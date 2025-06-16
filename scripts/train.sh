export PYTHONWARNINGS="ignore"

# single gpu
CUDA_VISIBLE_DEVICES="0," accelerate launch train.py \
--pretrained_model_name_or_path="preset/models/stable-diffusion-2-base" \
--seesr_model_path="preset/models/seesr" \
--tiny_vae_path="preset/models/tiny_vae" \
--output_dir="preset/train_output/DIV2K" \
--root_folders 'preset/datasets/train_datasets/DIV2K/paired_data' \
--ram_ft_path 'preset/models/DAPE.pth' \
--enable_xformers_memory_efficient_attention \
--mixed_precision="fp16" \
--resolution=512 \
--learning_rate=5e-6 \
--train_batch_size=1 \
--gradient_accumulation_steps=32 \
--max_train_steps=15000 \
--null_text_ratio=0.5 \
--dataloader_num_workers=0 \
--checkpointing_steps=1000 \
--gradient_checkpointing \
--use_8bit_adam \
--set_grads_to_none \
--allow_tf32 \