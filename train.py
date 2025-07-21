'''
 * SeeSR: Towards Semantics-Aware Real-World Image Super-Resolution 
 * Modified from Rongyuan Wu by Andrea Tomasoni
 * 03/07/2025
'''

import argparse
import logging
import math
import os
from pathlib import Path

import accelerate
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    AutoencoderTiny,
    DDPMScheduler,
)

from pipelines.pipeline_seesr import StableDiffusionControlNetPipeline
from ram.models.ram_lora import ram
from ram import inference_ram as inference

from models.controlnet import ControlNetModel
from models.unet_2d_condition import UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

from dataloaders.paired_dataset import PairedCaptionDataset

from torchvision import transforms
import torch.nn.functional as F

from utils_data.sam2_processing import load

from models.unet_2d_blocks import CrossAttnDownBlock2D, CrossAttnUpBlock2D, UNetMidBlock2DCrossAttn

import glob

import pyiqa

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.21.0.dev0")

logger = get_logger(__name__)

from torchvision import transforms
tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])
ram_transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def copy_dape_to_sam2_weights(model, accelerator):
    """
    Initializes ALL weights of the sam2 attention modules by copying or slicing 
    from the pre-trained dape (image_attentions) modules.
    """

    logger.info(f"Attempting to copy DAPE weights to SAM2 modules for {model.__class__.__name__}...")
    
    model_to_copy = accelerator.unwrap_model(model)
    
    def transfer_weights(source_attention_module, target_attention_module, block_name, target_name):
        source_sd = source_attention_module.state_dict()
        target_sd = target_attention_module.state_dict()
        
        new_target_sd = {}
        
        for key, target_param in target_sd.items():
            if key not in source_sd:
                logger.warning(f"  - Key '{key}' not found in source module. Keeping random init for {block_name}.{target_name}")
                new_target_sd[key] = target_param.clone()
                continue
            
            source_param = source_sd[key]
            
            # Shapes match, it's a direct copy
            if source_param.shape == target_param.shape:
                logger.info(f"  - Copying weight for '{key}' in {block_name}.{target_name}")
                new_target_sd[key] = source_param.clone()

            # Shapes mismatch, cross-attention projection layer.
            else:
                # Linear layer weights: [out_features, in_features]
                if target_param.dim() > 1 and source_param.shape[0] == target_param.shape[0]:
                    logger.info(f"  - Slicing weight for '{key}' in {block_name}.{target_name}")
                    slice_dim = target_param.shape[1]
                    new_target_sd[key] = source_param[:, :slice_dim].clone()

                # Bias terms
                elif target_param.dim() == 1 and source_param.shape[0] > target_param.shape[0]:
                     logger.warning(f"  - Mismatched bias term '{key}' in {block_name}.{target_name}. This is unusual.")
                     slice_dim = target_param.shape[0]
                     new_target_sd[key] = source_param[:slice_dim].clone()
                else:
                    logger.error(f"  - UNHANDLED MISMATCH for '{key}' in {block_name}.{target_name}. "
                                 f"Source: {source_param.shape}, Target: {target_param.shape}. Keeping random init.")
                    new_target_sd[key] = target_param.clone()

        target_attention_module.load_state_dict(new_target_sd)

    for block_name, block in model_to_copy.named_modules():
        if isinstance(block, (CrossAttnDownBlock2D, CrossAttnUpBlock2D, UNetMidBlock2DCrossAttn)) and hasattr(block, 'use_image_cross_attention') and block.use_image_cross_attention:
            logger.info(f"Processing block: {block_name}")
            
            if isinstance(block, UNetMidBlock2DCrossAttn):
                source_attns = block.attentions
                target_sam_image_attns = block.image_attentions
                target_sam_seg_attns = block.sam2_segmentation_attentions
            else:
                source_attns = block.image_attentions
                target_sam_image_attns = block.sam2_image_attentions
                target_sam_seg_attns = block.sam2_segmentation_attentions

            for i in range(len(source_attns)):
                source_module = source_attns[i]
                
                target_img_module = target_sam_image_attns[i]
                transfer_weights(source_module, target_img_module, block_name, f"sam2_image_attentions.{i}")
                
                target_seg_module = target_sam_seg_attns[i]
                transfer_weights(source_module, target_seg_module, block_name, f"sam2_segmentation_attentions.{i}")

    logger.info(f"Finished copying weights for {model.__class__.__name__}.")

def verify_weights(accelerator, model):

    if accelerator.is_main_process:
        logger.info("--- STARTING WEIGHT VERIFICATION ---")
        try:
            unwrapped_model = accelerator.unwrap_model(model)
            
            source_module = unwrapped_model.down_blocks[0].image_attentions[0].transformer_blocks[0].attn2
            target_module = unwrapped_model.down_blocks[0].sam2_image_attentions[0].transformer_blocks[0].attn2

            source_weight = source_module.to_k.weight.detach()
            target_dim = target_module.to_k.weight.shape[1]
            source_slice = source_weight[:, :target_dim]

            target_weight_after_copy = target_module.to_k.weight.detach()

            # Perform the check
            are_equal = torch.allclose(source_slice, target_weight_after_copy)
            
            logger.info(f"Source weight shape: {source_weight.shape}")
            logger.info(f"Target weight shape: {target_weight_after_copy.shape}")
            logger.info(f"Verification Check: Target weight successfully initialized from source slice? -> {are_equal}")

            if not are_equal:
                logger.error("VERIFICATION FAILED! Weights were not copied correctly.")
                diff = torch.abs(source_slice - target_weight_after_copy).max()
                logger.error(f"Max absolute difference: {diff.item()}")
            else:
                logger.info("VERIFICATION PASSED! Weights copied successfully.")

        except Exception as e:
            logger.error(f"An exception occurred during weight verification: {e}")
            logger.error("This might be due to an incorrect module path. Check your model structure.")
            
        logger.info("--- WEIGHT VERIFICATION COMPLETE ---")

def image_grid(imgs, rows, cols):

    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def log_validation(vae, text_encoder, unet, controlnet, args, accelerator, weight_dtype, validation_dataloader, tokenizer, global_step):

    logger.info("Running validation... ")
    logger.info("Generating validation image")
    if args.generate_validation_image:
        if accelerator.is_main_process:
            pipeline = StableDiffusionControlNetPipeline(
                vae=accelerator.unwrap_model(vae),
                text_encoder=accelerator.unwrap_model(text_encoder),
                tokenizer=tokenizer,
                unet=accelerator.unwrap_model(unet),
                controlnet=accelerator.unwrap_model(controlnet),
                scheduler=noise_scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=None
            )

            pipeline = pipeline.to(accelerator.device)
            pipeline.set_progress_bar_config(disable=True)

            ram_model = ram(pretrained='preset/models/ram_swin_large_14m.pth',
                            pretrained_condition=args.ram_ft_path,
                            image_size=384,
                            vit='swin_l')
            
            ram_model.eval()
            ram_model.to(accelerator.device)

            if args.validation_image and args.validation_prompt:
                validation_image_path = args.validation_image[0]
                validation_image = Image.open(validation_image_path).convert("RGB")

                lq_for_ram = tensor_transforms(validation_image).unsqueeze(0).to(accelerator.device)
                lq_for_ram = ram_transforms(lq_for_ram)
                ram_tags = inference(lq_for_ram, ram_model)
                ram_encoder_hidden_states = ram_model.generate_image_embeds(lq_for_ram)

                final_prompt = f"{ram_tags[0]}, "
                final_prompt += "clean, high-resolution, 8k,"
                negative_prompt = "dotted, noise, blur, lowres, smooth"

                width, height = validation_image.size
                cond_image = validation_image.resize((width*4, height*4))

                generator = torch.Generator(device=accelerator.device).manual_seed(args.seed if args.seed else 42)

                logger.info(f"Generating validation image for step {global_step} with prompt: '{final_prompt}'")
                
                with torch.autocast("cuda"):
                    generated_image = pipeline(
                        prompt=final_prompt,
                        image=cond_image,
                        negative_prompt=negative_prompt,
                        num_inference_steps=50,
                        generator=generator,
                        height=height*4,
                        width=width*4,
                        guidance_scale=5.5,
                        ram_encoder_hidden_states=ram_encoder_hidden_states,
                    ).images[0]

                # Create a directory for validation samples if it doesn't exist
                val_img_dir = os.path.join(args.output_dir, "validation_samples")
                os.makedirs(val_img_dir, exist_ok=True)
                
                # Save the generated image
                generated_image.save(os.path.join(val_img_dir, f"step_{global_step}.png"))
                logger.info(f"Saved validation image to {val_img_dir}/step_{global_step}.png")

            del pipeline
            torch.cuda.empty_cache()

    logger.info(f"Before validation: {unet.training=}, {controlnet.training=}")
    val_logs = {}

    try:

        unet.eval()
        controlnet.eval()

        logger.info(f"During validation: {unet.training=}, {controlnet.training=}")

        if validation_dataloader is not None:
            total_val_loss = 0.0
            total_val_diffusion_loss = 0.0
            total_val_semantic_loss = 0.0

            if args.use_lpips_loss:
                total_val_lpips_loss = 0.0
            
            for val_batch in tqdm(validation_dataloader, desc="Calculating validation loss", disable=not accelerator.is_local_main_process):
                with torch.no_grad():
                    pixel_values = val_batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    encoder_hidden_states = text_encoder(val_batch["input_ids"].to(accelerator.device))[0]
                    ram_encoder_hidden_states = val_batch["ram_values"].to(accelerator.device, dtype=weight_dtype)

                    controlnet_image = val_batch["conditioning_pixel_values"].to(accelerator.device, dtype=weight_dtype)

                    sam2_segmentation_encoder_hidden_states = val_batch["sam2_seg_embeds"]
                    sam2_encoder_hidden_states = val_batch["sam2_img_embeds"]
                    
                    if args.use_sam2:
                            
                        down_block_res_samples, mid_block_res_sample = controlnet(
                            noisy_latents, 
                            timesteps,
                            encoder_hidden_states=encoder_hidden_states, 
                            controlnet_cond=controlnet_image,
                            return_dict=False, 
                            image_encoder_hidden_states=ram_encoder_hidden_states,
                            sam2_encoder_hidden_states=sam2_encoder_hidden_states, 
                            sam2_segmentation_encoder_hidden_states=sam2_segmentation_encoder_hidden_states,
                        )
                        
                        model_pred = unet(
                            noisy_latents, 
                            timesteps, 
                            encoder_hidden_states=encoder_hidden_states,
                            down_block_additional_residuals=[sample.to(dtype=weight_dtype) for sample in down_block_res_samples],
                            mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                            image_encoder_hidden_states=ram_encoder_hidden_states,
                            sam2_encoder_hidden_states=sam2_encoder_hidden_states,
                            sam2_segmentation_encoder_hidden_states=sam2_segmentation_encoder_hidden_states
                        ).sample

                    else:

                        down_block_res_samples, mid_block_res_sample = controlnet(
                            noisy_latents, 
                            timesteps, 
                            encoder_hidden_states=encoder_hidden_states, 
                            controlnet_cond=controlnet_image,
                            return_dict=False, 
                            image_encoder_hidden_states=ram_encoder_hidden_states,
                        )
                        
                        model_pred = unet(
                            noisy_latents, 
                            timesteps, 
                            encoder_hidden_states=encoder_hidden_states,
                            down_block_additional_residuals=[sample.to(dtype=weight_dtype) for sample in down_block_res_samples],
                            mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                            image_encoder_hidden_states=ram_encoder_hidden_states,
                        ).sample

                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    else:
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    
                    diffusion_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    # diffusion_loss = F.l1_loss(model_pred.float(), target.float(), reduction="mean")
                    loss = diffusion_loss

                    if args.use_sam2_loss or args.use_lpips_loss:
                        alphas_cumprod = noise_scheduler.alphas_cumprod
                        alpha_bar_t = alphas_cumprod[timesteps]

                        while len(alpha_bar_t.shape) < len(noisy_latents.shape):
                            alpha_bar_t = alpha_bar_t.unsqueeze(-1)

                        pred_original_sample_latents = (noisy_latents - (1 - alpha_bar_t).sqrt() * model_pred) / alpha_bar_t.sqrt()
                        pred_image_latents_for_vae = pred_original_sample_latents / tiny_vae.config.scaling_factor
                        pred_image = tiny_vae.decode(pred_image_latents_for_vae.to(weight_dtype)).sample
                        sr_rgb = (pred_image.clamp(-1,1) + 1.0) / 2.0
                        
                        if args.use_sam2_loss:
                            sr_sam_input = sam2_norm(sr_rgb).to(accelerator.device)
                            sr_embeds = sam2_image_encoder(sr_sam_input)["vision_features"]

                            gt_rgb_normalized = (pixel_values.to(torch.float32) + 1.0) / 2.0
                            gt_sam_input = sam2_norm(gt_rgb_normalized).to(accelerator.device)
                            gt_embeds = sam2_image_encoder(gt_sam_input)["vision_features"]

                            semantic_loss = F.mse_loss(sr_embeds.float(), gt_embeds.float(), reduction="mean")

                            total_val_semantic_loss += semantic_loss.item()
                            loss += args.sam2_loss_weight * semantic_loss

                        if args.use_lpips_loss:
                            gt_rgb = (pixel_values.to(torch.float32) + 1.0) / 2.0
                            lpips_loss = lpips_metric(sr_rgb.to(torch.float32), gt_rgb)
                            total_val_lpips_loss += lpips_loss.item()
                            loss += args.lpips_loss_weight * lpips_loss

                    total_val_diffusion_loss += diffusion_loss.item()
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(validation_dataloader)
            avg_val_diffusion_loss = total_val_diffusion_loss / len(validation_dataloader)
            val_logs = {"val_loss": avg_val_loss, "val_diffusion_loss": avg_val_diffusion_loss}

            if args.use_sam2_loss:
                avg_val_semantic_loss = total_val_semantic_loss / len(validation_dataloader)
                val_logs["val_semantic_loss"] = avg_val_semantic_loss
            
            if args.use_lpips_loss:
                avg_val_lpips_loss = total_val_lpips_loss / len(validation_dataloader)
                val_logs["val_lpips_loss"] = avg_val_lpips_loss

    finally:
        torch.cuda.empty_cache()

        # Restore original training state of the models
        unet.train()
        controlnet.train()
        logger.info(f"After validation: {unet.training=}, {controlnet.training=}\n")

    return val_logs

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")

def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- controlnet
inference: true
---
    """
    model_card = f"""
# controlnet-{repo_id}

These are controlnet weights trained on {base_model} with new type of conditioning.
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/home/notebook/data/group/LowLevelLLM/models/diffusion_models/stable-diffusion-2-base",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--unet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained unet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--tiny_vae_path",
        type=str,
        default=None,
        help="Path to pretrained tiny vae.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./experience/test",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", 
        type=int, 
        default=16, 
        help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=int,
        default=1000
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int, 
        default=500, 
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power", 
        type=float, 
        default=1.0, 
        help="Power factor of the polynomial scheduler."
        )
    parser.add_argument(
        "--use_8bit_adam", 
        action="store_true", 
        help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1", 
        type=float, 
        default=0.9, 
        help="The beta1 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_beta2", 
        type=float, 
        default=0.999, 
        help="The beta2 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_weight_decay", 
        type=float, 
        default=1e-2, 
        help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon", 
        type=float, 
        default=1e-08,
        help="Epsilon value for the Adam optimizer"
    )
    parser.add_argument(
        "--max_grad_norm", 
        default=1.0, 
        type=float, 
        help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub", 
        action="store_true",
        help="Whether or not to push the model to the Hub."
    )
    parser.add_argument(
        "--hub_token", 
        type=str, 
        default=None, 
        help="The token to use to push to the Model Hub."
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", 
        action="store_true", 
        help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default='NOTHING',
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", 
        type=str, 
        default="image", 
        help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=[""],
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=[""],
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=1,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="SeeSR",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--root_folders", 
        type=str, 
        default=''
    )
    parser.add_argument(
        "--null_text_ratio", 
        type=float, 
        default=0.5
    )
    parser.add_argument(
        "--ram_ft_path", 
        type=str, 
        default=None
    )
    parser.add_argument(
        '--trainable_modules', 
        nargs='*', 
        type=str, 
        default=["image_attentions"]
    )
    parser.add_argument(
        "--train_controlnet_tag_attention",
        action="store_true",
        help="Set this flag to make the controlnet text (tag) attention modules trainable."
    )
    parser.add_argument(
        "--train_controlnet_dape_attention",
        action="store_true",
        help="Set this flag to make the controlnet image attention modules trainable."
    )
    parser.add_argument(
        "--train_unet_tag_attention",
        action="store_true",
        help="Set this flag to make the unet text (tag) attention modules trainable."
    )
    parser.add_argument(
        "--train_unet_dape_attention",
        action="store_true",
        help="Set this flag to make the unet image attention modules trainable."
    )
    parser.add_argument(
        "--train_controlnet_sam2_attention",
        action="store_true",
        help="Set this flag to make the controlnet sam2 attention modules trainable."
    )
    parser.add_argument(
        "--train_unet_sam2_attention",
        action="store_true",
        help="Set this flag to make the unet sam2 attention modules trainable."
    )
    parser.add_argument(
        "--train_controlnet_fusion_conv",
        action="store_true",
        help="Set this flag to enable the merging of the embeddings using convolution in the ControlNet."
    )
    parser.add_argument(
        "--train_unet_fusion_conv",
        action="store_true",
        help="Set this flag to enable the merging of the embeddings using convolution in the UNet."
    )
    parser.add_argument(
        "--train_controlnet_double_fusion_conv",
        action="store_true",
        help="Set this flag to enable the merging of the embeddings using convolution->leaky_relu->convolution in the ControlNet."
    )
    parser.add_argument(
        "--train_unet_double_fusion_conv",
        action="store_true",
        help="Set this flag to enable the merging of the embeddings using convolution->leaky_relu->convolution in the UNet."
    )
    parser.add_argument(
        "--finetune_lr",
        type=float,
        default=5e-4,
        help="Initial learning rate for newly added modules. Should typically be higher."
    )
    parser.add_argument(
        "--seesr_model_path", 
        type=str, 
        default=None
    )
    parser.add_argument(
        "--use_sam2_loss",
        action="store_true",
        help="Whether to use SAM2 to compute the perceptual loss."
    )
    parser.add_argument(
        "--sam2_loss_weight", 
        type=float, 
        default=1
    )
    parser.add_argument(
        "--use_sam2", 
        action="store_true", 
        help="Whether to use SAM2 for image conditioning."
    )
    parser.add_argument(
        "--use_lpips_loss",
        action="store_true",
        help="Whether to use LPIPS loss for training."
    )
    parser.add_argument(
        "--lpips_loss_weight",
        type=float,
        default=1.0,
        help="Weight for the LPIPS loss in the total loss calculation."
    )
    parser.add_argument(
        "--validation_data_dir",
        type=str,
        default=None,
        help="A folder containing the validation data, with the same structure as the training data.",
    )
    parser.add_argument(
        "--finetune_from_checkpoint",
        type=str,
        default=None,
        help="Path to a full checkpoint directory (containing models, optimizer, scheduler) to start finetuning from."
    )
    parser.add_argument(
        "--generate_validation_image",
        action="store_true",
        help="Wether to generate the image during validation"
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    if args.dataset_name is not None and args.train_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and len(args.validation_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args

args = parse_args()
logging_dir = Path(args.output_dir, args.logging_dir)

from accelerate import DistributedDataParallelKwargs
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

accelerator = Accelerator(
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    mixed_precision=args.mixed_precision,
    log_with=args.report_to,
    project_config=accelerator_project_config,
    kwargs_handlers=[ddp_kwargs]
)

## Make one log on every process with the configuration for debugging.
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger.info(accelerator.state, main_process_only=False)
if accelerator.is_local_main_process:
    transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_info()
else:
    transformers.utils.logging.set_verbosity_error()
    diffusers.utils.logging.set_verbosity_error()

# Silence SAM 2 warnings
sam_logger = logging.getLogger("root")
sam_logger.setLevel(logging.WARNING)
logger.setLevel(logging.INFO)

# If passed along, set the training seed now.
if args.seed is not None:
    set_seed(args.seed)

# Handle the repository creation
if accelerator.is_main_process:
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.push_to_hub:
        repo_id = create_repo(
            repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
        ).repo_id

# Load the tokenizer
if args.tokenizer_name:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
elif args.pretrained_model_name_or_path:
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )

text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

# Load scheduler and models
noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.to(accelerator.device)

# Text encoder
text_encoder = text_encoder_cls.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, device_map={"": str(accelerator.device)})

# VAE
vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
vae.enable_tiling()
vae.enable_slicing()

# Tiny VAE (used for the loss)
tiny_vae = AutoencoderTiny.from_pretrained(args.tiny_vae_path)
tiny_vae.enable_tiling()
tiny_vae.enable_slicing()

# SAM 2.1
if args.use_sam2 or args.use_sam2_loss:
    sam2 = load(apply_postprocessing=False, stability_score_thresh=0.5)
    sam2_image_encoder = sam2.predictor.model.image_encoder
    sam2_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

if args.use_lpips_loss:
    logger.info("Loading LPIPS loss model")
    lpips_metric = pyiqa.create_metric("lpips", device=accelerator.device, as_loss=True)

if args.unet_model_name_or_path:
    # resume from self-train
    logger.info("Loading unet weights from self-train")

    unet = UNet2DConditionModel.from_pretrained_orig(
        args.pretrained_model_name_or_path, 
        args.unet_model_name_or_path, 
        subfolder="unet", 
        revision=args.revision, 
        use_image_cross_attention=True,
    )

    print(f'===== if use ram encoder? {unet.config.use_image_cross_attention}')

else:
    # resume from pretrained SD
    logger.info("Loading unet weights from SeeSR")

    unet = UNet2DConditionModel.from_pretrained(
        args.seesr_model_path, 
        subfolder="unet", 
        use_image_cross_attention=True, 
        device_map=None, 
        low_cpu_mem_usage=False,
        revision=args.revision
    )
    
    print(f'===== if use ram encoder? {unet.config.use_image_cross_attention}')

if args.controlnet_model_name_or_path:
    # resume from self-train
    logger.info("Loading existing controlnet weights")

    controlnet = ControlNetModel.from_pretrained(
        args.controlnet_model_name_or_path, 
        subfolder="controlnet", 
        use_image_cross_attention=True, 
        device_map=None, 
        low_cpu_mem_usage=False
    )

else:
    logger.info("Initializing controlnet weights from unet")
    
    controlnet = ControlNetModel.from_unet(
        unet, 
        use_image_cross_attention=True
    )
    
# `accelerate` 0.16.0 will have better support for customized saving
if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        i = len(weights) - 1
        assert len(models) == 2 and len(weights) == 2
        for i, model in enumerate(models):
            sub_dir = "unet" if isinstance(model, UNet2DConditionModel) else "controlnet"
            model.save_pretrained(os.path.join(output_dir, sub_dir))
            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()

    def load_model_hook(models, input_dir):
        assert len(models) == 2
        for i in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            if not isinstance(model, UNet2DConditionModel):
                load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet") # , low_cpu_mem_usage=False, ignore_mismatched_sizes=True
            else:
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet") # , low_cpu_mem_usage=False, ignore_mismatched_sizes=True

            model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

text_encoder.eval()
text_encoder.requires_grad_(False)
vae.eval()
vae.requires_grad_(False)
tiny_vae.eval()
tiny_vae.requires_grad_(False)

if args.use_sam2 or args.use_sam2_loss:
    sam2_image_encoder.eval()
    sam2_image_encoder.requires_grad_(False)

controlnet.train()
controlnet.requires_grad_(False)
unet.train()
unet.requires_grad_(False)

# Make ControlNet and UNet modules trainable
if args.train_controlnet_tag_attention:
    trainable_module_found = False

    for name, module in controlnet.named_modules():
        if name.endswith("attentions") and "image" not in name and "sam2" not in name:
            for params in module.parameters():
                params.requires_grad = True

            if not trainable_module_found:
                print("Found trainable modules in ControlNet:")
            print(f'  - {name}')
            trainable_module_found = True

if args.train_controlnet_dape_attention:
    trainable_module_found = False

    for name, module in controlnet.named_modules():
        if name.endswith("image_attentions") and "sam2" not in name:
            for params in module.parameters():
                params.requires_grad = True

            if not trainable_module_found:
                print("Found trainable modules in ControlNet:")
            print(f'  - {name}')
            trainable_module_found = True

if args.train_unet_tag_attention:
    trainable_module_found = False

    for name, module in unet.named_modules():
        if name.endswith("attentions") and "image" not in name and "sam2" not in name:
            for params in module.parameters():
                params.requires_grad = True

            if not trainable_module_found:
                print("Found trainable modules in UNet:")   
            print(f'  - {name}')
            trainable_module_found = True

if args.train_unet_dape_attention:
    trainable_module_found = False

    for name, module in unet.named_modules():
        if name.endswith("image_attentions") and "sam2" not in name:
            for params in module.parameters():
                params.requires_grad = True

            if not trainable_module_found:
                print("Found trainable modules in UNet:")
            print(f'  - {name}')
            trainable_module_found = True

if args.train_controlnet_sam2_attention:
    trainable_module_found = False

    for name, module in controlnet.named_modules():
        if "sam2" in name:
            for params in module.parameters():
                params.requires_grad = True

            if not trainable_module_found:
                print("Found trainable modules in UNet:")
            print(f'  - {name}')
            trainable_module_found = True
            
if args.train_unet_sam2_attention:
    trainable_module_found = False

    for name, module in unet.named_modules():
        if "sam2" in name:
            for params in module.parameters():
                params.requires_grad = True

            if not trainable_module_found:
                print("Found trainable modules in UNet:")
            print(f'  - {name}')
            trainable_module_found = True

if args.train_controlnet_fusion_conv:
    trainable_module_found = False

    for name, module in controlnet.named_modules():
        if name.endswith("fusion_conv"):
            for params in module.parameters():
                params.requires_grad = True

            if not trainable_module_found:
                print("Found trainable modules in ControlNet:")
            print(f'  - {name}')
            trainable_module_found = True

if args.train_controlnet_double_fusion_conv:
    trainable_module_found = False

    for name, module in controlnet.named_modules():
        if name.endswith("double_fusion_conv"):
            for params in module.parameters():
                params.requires_grad = True

            if not trainable_module_found:
                print("Found trainable modules in ControlNet:")
            print(f'  - {name}')
            trainable_module_found = True

if args.train_unet_fusion_conv:
    trainable_module_found = False

    for name, module in unet.named_modules():
        if name.endswith("fusion_conv"):
            for params in module.parameters():
                params.requires_grad = True

            if not trainable_module_found:
                print("Found trainable modules in UNet:")
            print(f'  - {name}')
            trainable_module_found = True

if args.train_unet_double_fusion_conv:
    trainable_module_found = False

    for name, module in unet.named_modules():
        if name.endswith("double_fusion_conv"):
            for params in module.parameters():
                params.requires_grad = True

            if not trainable_module_found:
                print("Found trainable modules in UNet:")
            print(f'  - {name}')
            trainable_module_found = True

if args.enable_xformers_memory_efficient_attention:
    if is_xformers_available():
        import xformers

        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            logger.warn(
                "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
            )
        unet.enable_xformers_memory_efficient_attention()
        controlnet.enable_xformers_memory_efficient_attention()
    else:
        raise ValueError("xformers is not available. Make sure it is installed correctly")

if args.gradient_checkpointing:
    logger.info("Enabling gradient checkpointing.")

    unet.enable_gradient_checkpointing()
    controlnet.enable_gradient_checkpointing()

# Check that all trainable models are in full precision
low_precision_error_string = (
    " Please make sure to always have all model weights in full float32 precision when starting training - even if"
    " doing mixed precision training, copy of the weights should still be float32."
)

if accelerator.unwrap_model(controlnet).dtype != torch.float32:
    raise ValueError(
        f"Controlnet loaded as datatype {accelerator.unwrap_model(controlnet).dtype}. {low_precision_error_string}"
    )
if accelerator.unwrap_model(unet).dtype != torch.float32:
    raise ValueError(
        f"Unet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
    )

# Enable TF32 for faster training on Ampere GPUs,
# cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
if args.allow_tf32:
    torch.backends.cuda.matmul.allow_tf32 = True

if args.scale_lr:
    args.learning_rate = (
        args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
    )

# Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
if args.use_8bit_adam:
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise ImportError(
            "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
        )

    optimizer_class = bnb.optim.AdamW8bit
else:
    optimizer_class = torch.optim.AdamW

# Optimizer creation
# print(f'================= Optimize ControlNet and Unet ======================')

# total_individual_params = sum(p.numel() for p in (list(controlnet.parameters()) + list(unet.parameters())))
# params_to_optimize = [p for p in controlnet.parameters() if p.requires_grad] + [p for p in unet.parameters() if p.requires_grad]
# trainable_tensors = len(params_to_optimize)
# trainable_individual_params = sum(p.numel() for p in params_to_optimize)

# print(f"Total Individual Parameters: {total_individual_params:,}")
# print(f"Trainable (ControlNet + UNet) Parameters: {trainable_individual_params:,}\n")

# print(f'start to load optimizer...')

# # TODO: cambia optimizer e usa SGD
# # lr non cambiare
# optimizer = optimizer_class(
#     params_to_optimize,
#     lr=args.learning_rate,
#     betas=(args.adam_beta1, args.adam_beta2),
#     weight_decay=args.adam_weight_decay,
#     eps=args.adam_epsilon,
# )

# Optimizer creation
print(f'================= Optimize ControlNet and Unet ======================')
print(f'start to load optimizer...')

new_module_params = []
base_model_params = []
new_module_keywords = ["fusion_conv"]

params_to_optimize_full_list = list(controlnet.named_parameters()) + list(unet.named_parameters())

for name, param in params_to_optimize_full_list:
    if param.requires_grad:
        if any(keyword in name for keyword in new_module_keywords):
            new_module_params.append(param)
        else:
            base_model_params.append(param)

num_new_params = sum(p.numel() for p in new_module_params)
num_base_params = sum(p.numel() for p in base_model_params)
print(f"Trainable Base Model Parameters: {num_base_params:,} (LR: {args.learning_rate})")
print(f"Trainable New Module Parameters: {num_new_params:,} (LR: {args.finetune_lr})")
print(f"Total Trainable Parameters: {num_base_params + num_new_params:,}\n")

optimizer_grouped_parameters = [
    {
        "params": base_model_params,
        "lr": args.learning_rate,
    },
    {
        "params": new_module_params,
        "lr": args.finetune_lr,
    },
]

optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if g['params']]

if not optimizer_grouped_parameters:
    raise ValueError("No trainable parameters found. Check your --train_* flags.")

optimizer = optimizer_class(
    optimizer_grouped_parameters,
    betas=(args.adam_beta1, args.adam_beta2),
    weight_decay=args.adam_weight_decay,
    eps=args.adam_epsilon,
)

params_to_optimize = base_model_params + new_module_params

logger.info("Creating train dataloader...")
train_dataset = PairedCaptionDataset(
    root_folders=args.root_folders,
    tokenizer=tokenizer,
    null_text_ratio=args.null_text_ratio,
    validation=False,
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    num_workers=args.dataloader_num_workers,
    batch_size=args.train_batch_size,
    shuffle=True,
)

logger.info("Creating validation dataloader...")
validation_dataset = PairedCaptionDataset(
    root_folders=args.root_folders,
    tokenizer=tokenizer,
    null_text_ratio=0.0,
    validation=True,
)

validation_dataloader = torch.utils.data.DataLoader(
    validation_dataset,
    num_workers=args.dataloader_num_workers,
    batch_size=args.train_batch_size,
    shuffle=False,
)

# Scheduler and math around the number of training steps.
overrode_max_train_steps = False
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
if args.max_train_steps is None:
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    overrode_max_train_steps = True

lr_scheduler = get_scheduler(
    args.lr_scheduler,
    optimizer=optimizer,
    num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
    num_training_steps=args.max_train_steps * accelerator.num_processes,
    num_cycles=args.lr_num_cycles,
    power=args.lr_power,
)

# Prepare everything with our `accelerator`.
if validation_dataloader:
    controlnet, unet, optimizer, train_dataloader, lr_scheduler, validation_dataloader = accelerator.prepare(
        controlnet, unet, optimizer, train_dataloader, lr_scheduler, validation_dataloader
    )
else:
    controlnet, unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, unet, optimizer, train_dataloader, lr_scheduler
    )

# if args.use_sam2:
#     # TODO: verify that is correct
#     copy_dape_to_sam2_weights(unet, accelerator)
#     copy_dape_to_sam2_weights(controlnet, accelerator)

#     verify_weights(accelerator, controlnet)
#     verify_weights(accelerator, unet)

# For mixed precision training we cast the text_encoder and vae weights to half-precision
# as these models are only used for inference, keeping weights in full precision is not required.

weight_dtype = torch.float32
if accelerator.mixed_precision == "fp16":
    weight_dtype = torch.float16
elif accelerator.mixed_precision == "bf16":
    weight_dtype = torch.bfloat16

vae.to(accelerator.device, dtype=weight_dtype)
tiny_vae.to(accelerator.device, dtype=weight_dtype)

if args.use_sam2 or args.use_sam2_loss:
    sam2_image_encoder.to(accelerator.device, dtype=weight_dtype)

num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
if overrode_max_train_steps:
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

# Afterwards we recalculate our number of training epochs
args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

# We need to initialize the trackers we use, and also store our configuration.
# The trackers initializes automatically on the main process.
if accelerator.is_main_process:
    tracker_config = dict(vars(args))

    # tensorboard cannot handle list types for config
    tracker_config.pop("validation_prompt")
    tracker_config.pop("validation_image")
    tracker_config.pop("trainable_modules")

    accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

# Train!
total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

logger.info("***** Running training *****")
logger.info(f"  Num examples = {len(train_dataset)}")
logger.info(f"  Num batches each epoch = {len(train_dataloader)}")

logger.info(f"  Num Epochs = {args.num_train_epochs}")
logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
logger.info(f"  Total optimization steps = {args.max_train_steps}")
global_step = 0
first_epoch = 0

# Potentially load in the weights and states from a previous save
if args.resume_from_checkpoint:
    if args.resume_from_checkpoint != "latest":
        path = os.path.basename(args.resume_from_checkpoint)
    else:
        # Get the most recent checkpoint
        dirs = os.listdir(args.output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None

    if path is None:
        accelerator.print(
            f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
        )
        args.resume_from_checkpoint = None
        initial_global_step = 0
    else:
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(args.output_dir, path))
        global_step = int(path.split("-")[1])

        initial_global_step = global_step
        first_epoch = global_step // num_update_steps_per_epoch
else:
    initial_global_step = 0

progress_bar = tqdm(
    range(0, args.max_train_steps),
    initial=initial_global_step,
    desc="Steps",
    # Only show the progress bar once on each machine.
    disable=not accelerator.is_local_main_process,
)

image_logs = None

# MAIN TRAINING LOOP
for epoch in range(first_epoch, args.num_train_epochs):
    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(controlnet), accelerator.accumulate(unet):

            # Ground truth image (512x512)
            pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)

            # Convert GT image to latent space
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Sample noise to be added to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep (FORWARD DIFFUSION PROCESS (Z_t in the paper))
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Compute the embeddings for the tags (TEXT EMBEDDING BRANCH)
            encoder_hidden_states = text_encoder(batch["input_ids"].to(accelerator.device))[0]

            # Low resolution image for the controlnet (IMAGE BRANCH) -> the image encoder drawn in the paper is actually inside the controlnet
            controlnet_image = batch["conditioning_pixel_values"].to(accelerator.device, dtype=weight_dtype)

            with torch.no_grad():

                # Low resolution image that has been bicubic upsampled for the image encoder inside DAPE (384x384)
                ram_encoder_hidden_states = batch["ram_values"].to(accelerator.device, dtype=weight_dtype)

                # SAM2 Segmentation + image embeddings
                sam2_segmentation_encoder_hidden_states = batch["sam2_seg_embeds"].to(accelerator.device, dtype=weight_dtype)
                sam2_encoder_hidden_states = batch["sam2_img_embeds"].to(accelerator.device, dtype=weight_dtype)

            if args.use_sam2:
                # ControlNet
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                    image_encoder_hidden_states=ram_encoder_hidden_states,
                    sam2_encoder_hidden_states=sam2_encoder_hidden_states,
                    sam2_segmentation_encoder_hidden_states=sam2_segmentation_encoder_hidden_states,
                )

                # UNet
                # model_pred [1, 4, 64, 64]
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    image_encoder_hidden_states=ram_encoder_hidden_states,
                    sam2_encoder_hidden_states=sam2_encoder_hidden_states,
                    sam2_segmentation_encoder_hidden_states=sam2_segmentation_encoder_hidden_states
                ).sample

            else:
                # ControlNet
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                    image_encoder_hidden_states=ram_encoder_hidden_states,
                )

                # UNet
                # model_pred [1, 4, 64, 64]
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    image_encoder_hidden_states=ram_encoder_hidden_states,
                ).sample

            ######################################################################################################
            # --- COMPUTATIONAL GRAPH DEBUG CHECK 1 ---
            if step == 1:
                print("\n--- model_pred.grad_fn Check: Step 1 ---")
                print(f"    model_pred.requires_grad: {model_pred.requires_grad}")
                print(f"    model_pred.is_leaf: {model_pred.is_leaf}")
                print(f"    model_pred.grad_fn: {model_pred.grad_fn}")

                if model_pred.grad_fn is None:
                    print("    CRITICAL: Graph broken at model output! Check UNet/ControlNet internals.")
            ######################################################################################################
                
            del encoder_hidden_states, controlnet_image, ram_encoder_hidden_states, sam2_encoder_hidden_states, sam2_segmentation_encoder_hidden_states
            torch.cuda.empty_cache()

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
            if args.use_sam2_loss or args.use_lpips_loss:
                # Take the latent code of the GT image and convert it to the rgb image
                ######################################################################################################
                alphas_cumprod = noise_scheduler.alphas_cumprod
                
                # Get the alpha term for the current timestep t
                alpha_bar_t = alphas_cumprod[timesteps]
                
                # Reshape for broadcasting: from [B] to [B, 1, 1, 1]
                while len(alpha_bar_t.shape) < len(noisy_latents.shape):
                    alpha_bar_t = alpha_bar_t.unsqueeze(-1)

                # Predict x0 from xt and epsilon
                pred_original_sample_latents = (noisy_latents - (1 - alpha_bar_t).sqrt() * model_pred) / alpha_bar_t.sqrt()
                
                # Now, decode the predicted original sample as before
                pred_image_latents_for_vae = pred_original_sample_latents / tiny_vae.config.scaling_factor
                pred_image = tiny_vae.decode(pred_image_latents_for_vae.to(weight_dtype)).sample
                sr_rgb = (pred_image.clamp(-1,1) + 1.0) / 2.0

                ######################################################################################################
                # --- COMPUTATIONAL GRAPH DEBUG CHECK 2 ---
                if step == 1:
                    print("\n--- sr_rgb.grad_fn Check: Step 2 ---")
                    print(f"    sr_rgb.grad_fn: {sr_rgb.grad_fn}")

                    if sr_rgb.grad_fn is None:
                        print("    CRITICAL: Graph broken during sr_rgb derivation! Check VAE decode or math.")

                ######################################################################################################

                if args.use_sam2_loss:
                    sr_sam_input = sam2_norm(sr_rgb).to(accelerator.device)
                    sr_embeds = sam2_image_encoder(sr_sam_input)["vision_features"]

                    with torch.no_grad():
                        gt_sam_input = sam2_norm(pixel_values).to(accelerator.device)
                        gt_embeds = sam2_image_encoder(gt_sam_input)["vision_features"]

                    semantic_loss = F.mse_loss(sr_embeds.float(), gt_embeds.float(), reduction="mean")

                    ######################################################################################################
                    # --- COMPUTATIONAL GRAPH DEBUG CHECK 3 ---
                    if step == 1:
                        print("\n--- Semantic Loss grad_fn Check: Step 3 ---")
                        print(f"    sr_rgb.grad_fn: {sr_rgb.grad_fn}")
                        print(f"    sr_sam_input.grad_fn: {sr_sam_input.grad_fn}")
                        print(f"    sr_embeds.grad_fn: {sr_embeds.grad_fn}")
                        print(f"    semantic_loss.grad_fn: {semantic_loss.grad_fn}")
                        print(f"    semantic_loss.requires_grad: {semantic_loss.requires_grad}")
                        print("-" * 20)
                    
                    ######################################################################################################
                if args.use_lpips_loss:
                    gt_rgb = (pixel_values.to(torch.float32) + 1.0) / 2.0
                    lpips_loss = lpips_metric(sr_rgb.to(torch.float32), gt_rgb)

            diffusion_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            # diffusion_loss = F.l1_loss(model_pred.float(), target.float(), reduction="mean")

            ######################################################################################################
            # --- COMPUTATIONAL GRAPH DEBUG CHECK 4 ---
            if step == 1:
                print("\n--- Dissusion Loss grad_fn Check: Step 4 ---")
                print(f"    diffusion_loss.requires_grad: {diffusion_loss.requires_grad}")
                print(f"    diffusion_loss.grad_fn: {diffusion_loss.grad_fn}")
                print("-" * 20)
            
            ######################################################################################################

            if args.use_sam2_loss:
                loss = diffusion_loss + args.sam2_loss_weight * semantic_loss

            elif args.use_lpips_loss:
                loss = diffusion_loss + args.lpips_loss_weight * lpips_loss

            else:
                loss = diffusion_loss
            
            ######################################################################################################
            # --- COMPUTATIONAL GRAPH DEBUG CHECK 5 ---
            if step == 1:
                print("\n--- Main Loss grad_fn Check: Step 5 ---")
                print(f"    loss.grad_fn: {loss.grad_fn}")

                if loss.grad_fn is None:
                    print("    CRITICAL: Graph broken at final loss calculation!")

                print("-" * 20 + "\n")
            ######################################################################################################

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                
                total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2).cpu() for p in params_to_optimize if p.grad is not None]), 2)
                accelerator.log({"grad_norm": total_norm.item()}, step=global_step)

                accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

        validation_loss = None
        
        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1

            if accelerator.is_main_process:
                if global_step % args.checkpointing_steps == 0:

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

                if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                    val_logs = log_validation(
                        vae,
                        text_encoder,
                        unet,
                        controlnet,
                        args,
                        accelerator,
                        weight_dtype,
                        validation_dataloader,
                        tokenizer,
                        global_step,
                    )

                    validation_loss = val_logs["val_loss"]

        if args.use_sam2_loss:
            logs = {"loss/train": loss.detach().item(), "loss/train_diffusion": diffusion_loss.detach().item(), "loss/train_semantic": semantic_loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}

            if validation_loss is not None:
                logger.info(f"validation_loss: {validation_loss:.4f}")
                logs["loss/validation"] = validation_loss

        elif args.use_lpips_loss:
            logs = {"loss/train": loss.detach().item(), "loss/train_diffusion": diffusion_loss.detach().item(), "loss/train_lpips": lpips_loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}

            if validation_loss is not None:
                logger.info(f"validation_loss: {validation_loss:.4f}")
                logs["loss/validation"] = validation_loss
                
        else:
            logs = {"loss/train": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}

            if validation_loss is not None:
                logs["loss/validation"] = validation_loss
                logger.info(f"validation_loss: {validation_loss:.4f}")

        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)

        if global_step >= args.max_train_steps:
            break

# Create the pipeline using the trained modules and save it.
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    controlnet = accelerator.unwrap_model(controlnet)
    controlnet.save_pretrained(args.output_dir)

    unet = accelerator.unwrap_model(unet)
    unet.save_pretrained(args.output_dir)

    if args.push_to_hub:
        save_model_card(
            repo_id,
            image_logs=image_logs,
            base_model=args.pretrained_model_name_or_path,
            repo_folder=args.output_dir,
        )
        upload_folder(
            repo_id=repo_id,
            folder_path=args.output_dir,
            commit_message="End of training",
            ignore_patterns=["step_*", "epoch_*"],
        )

    validation_images_list = glob.glob(os.path.join(args.output_dir + "/validation_samples", "*.png"))
    validation_images_list.sort(key=lambda f: int(os.path.basename(f).split('_')[1].split('.')[0]))
    validation_images = []
    for path in validation_images_list:
        img = Image.open(path).convert("RGB")
        validation_images.append(img)

    image_grid(validation_images, 1, len(validation_images)).save(os.path.join(args.output_dir, "validation_samples/grid.png"))

accelerator.end_training()
