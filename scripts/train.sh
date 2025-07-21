#!/bin/bash

# Paths
PRETRAINED_MODEL_PATH="preset/models/stable-diffusion-2-base"
SEESR_MODEL_PATH="preset/models/seesr"
TINY_VAE_PATH="preset/models/tiny_vae"
OUTPUT_DIR="preset/train_output/LSDIR"
ROOT_FOLDERS="preset/datasets/train_datasets/LSDIR"
RAM_FT_PATH="preset/models/DAPE.pth"
VALIDATION_PATH="preset/datasets/test_datasets/RealSR/test_LR/Canon_004_LR4.png"

# Hyperparameters
LEARNING_RATE=5e-6
FINETUNE_LR=5e-5
TRAIN_BATCH_SIZE=1
GRAD_ACCUM_STEPS=4
MAX_TRAIN_STEPS=7500
CHECKPOINTING_STEPS=500
NULL_TEXT_RATIO=0.5
RESOLUTION=512
DATALOADER_WORKERS=0
CUDA_DEVICES="0"
SAM_LOSS_WEIGHT=1
LR_WARMUP_STEPS=500
LR_SCHEDULER="constant_with_warmup"
VALIDATION_STEPS=250
VALIDATION_PROMPT=""
PRECISION="bf16"
RESUME_CHECKPOINT="latest"

export PYTHONWARNINGS="ignore"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"

echo "Starting training process..."
echo "Output directory: $OUTPUT_DIR"
echo "Checkpoints will be saved every $CHECKPOINTING_STEPS steps."

while true
do
    accelerate launch train.py \
      --pretrained_model_name_or_path="$PRETRAINED_MODEL_PATH" \
      --controlnet_model_name_or_path="$SEESR_MODEL_PATH" \
      --unet_model_name_or_path="$SEESR_MODEL_PATH" \
      --seesr_model_path="$SEESR_MODEL_PATH" \
      --tiny_vae_path="$TINY_VAE_PATH" \
      --output_dir="$OUTPUT_DIR" \
      --root_folders="$ROOT_FOLDERS" \
      --ram_ft_path="$RAM_FT_PATH" \
      --enable_xformers_memory_efficient_attention \
      --mixed_precision="$PRECISION" \
      --resolution=$RESOLUTION \
      --learning_rate=$LEARNING_RATE \
      --finetune_lr=$FINETUNE_LR \
      --train_batch_size=$TRAIN_BATCH_SIZE \
      --gradient_accumulation_steps=$GRAD_ACCUM_STEPS \
      --max_train_steps=$MAX_TRAIN_STEPS \
      --null_text_ratio=$NULL_TEXT_RATIO \
      --dataloader_num_workers=$DATALOADER_WORKERS \
      --checkpointing_steps=$CHECKPOINTING_STEPS \
      --gradient_checkpointing \
      --use_8bit_adam \
      --allow_tf32 \
      --lr_warmup_steps=$LR_WARMUP_STEPS \
      --lr_scheduler=$LR_SCHEDULER \
      --validation_steps=$VALIDATION_STEPS \
      --validation_image="$VALIDATION_PATH" \
      --validation_prompt="$VALIDATION_PROMPT" \
      --generate_validation_image \
      --resume_from_checkpoint="$RESUME_CHECKPOINT" \
      --train_controlnet_dape_attention \
      --train_controlnet_fusion_conv \
      --train_unet_dape_attention \
      --train_unet_fusion_conv \
      --use_sam2 \

    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "-----------------------------------"
        echo "Training completed successfully."
        echo "-----------------------------------"
        break
    else
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "  Training script failed with exit code $EXIT_CODE."
        echo "  Restarting from the latest checkpoint in 15 seconds."
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        sleep 15 # Wait for a moment before restarting to avoid rapid-fire crashes
    fi
done