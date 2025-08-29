#!/bin/bash

datasets=("DIV2K" "DRealSR" "RealLR200" "RealSR")
# datasets=("RealSR")
# datasets=("RealLR200")

GENERATION_SCRIPT="test_seesr.py"
METRIC_SCRIPT="utils_data/metrics.py"
SKIP_GENERATION=false

# Paths
PRETRAINED_MODEL_PATH="preset/models/stable-diffusion-2-base"
RAM_FT_PATH="preset/models/DAPE.pth"
CHECKPOINT_BASE_DIR="preset/train_output/LSDIR/Final_long_training_all_sam2_sam2loss_ffhq"

# Directories
BASE_TEST_DATA_DIR="preset/datasets/test_datasets"
BASE_OUTPUT_DIR="preset/datasets/output"

# Hyperparameters
PROMPT=""
START_POINT="lr"
NUM_INFERENCE_STEPS=50
GUIDANCE_SCALE=5.5
IMAGE_SIZE=512
PRECISION="bf16"

mkdir -p "$BASE_OUTPUT_DIR"
for checkpoint_path in $(find "$CHECKPOINT_BASE_DIR" -maxdepth 1 -type d -name "checkpoint-*" | sort -V); do

    if [ ! -d "$checkpoint_path" ]; then
        echo "WARNING: No checkpoint found matching pattern in '$CHECKPOINT_BASE_DIR'. Exiting."
        continue
    fi

    FINETUNED_MODEL_PATH="$checkpoint_path"
    MODEL_NAME=$(basename "$checkpoint_path")

    echo "======================================================================"
    echo "## PROCESSING CHECKPOINT: $MODEL_NAME"
    echo "======================================================================"

    for dataset_name in "${datasets[@]}"; do
        echo "######################################################################"
        echo "## PROCESSING DATASET: $dataset_name for CHECKPOINT: $MODEL_NAME"
        echo "######################################################################"
        echo

        LR_IMAGE_PATH="${BASE_TEST_DATA_DIR}/${dataset_name}/test_LR"
        GT_PATH="${BASE_TEST_DATA_DIR}/${dataset_name}/test_HR"
        SR_OUTPUT_PATH="${BASE_OUTPUT_DIR}/${MODEL_NAME}/${dataset_name}"

        if [ "$SKIP_GENERATION" = false ]; then
            echo "--- Step 1: Generating images for ${dataset_name} ---"

            if [ ! -d "$LR_IMAGE_PATH" ]; then
                echo "ERROR: LR image directory not found at '$LR_IMAGE_PATH'. Skipping dataset."
                echo
                continue
            fi

            mkdir -p "$SR_OUTPUT_PATH"
            
            echo "Using finetuned model from: $FINETUNED_MODEL_PATH"
            echo "Saving generated images to: $SR_OUTPUT_PATH"
            
            python -W ignore "$GENERATION_SCRIPT" \
                --pretrained_model_path="$PRETRAINED_MODEL_PATH" \
                --finetuned_model_path="$FINETUNED_MODEL_PATH" \
                --ram_ft_path="$RAM_FT_PATH" \
                --image_path="$LR_IMAGE_PATH" \
                --output_dir="$SR_OUTPUT_PATH" \
                --prompt="$PROMPT" \
                --start_point="$START_POINT" \
                --num_inference_steps=$NUM_INFERENCE_STEPS \
                --guidance_scale=$GUIDANCE_SCALE \
                --process_size=$IMAGE_SIZE \
                --mixed_precision="$PRECISION"
            
            echo "--- Image generation for ${dataset_name} complete. ---"
        else
            echo "--- Step 1: SKIPPING image generation for ${dataset_name} as configured. ---"
        fi

        echo
        echo "--- Step 2: Calculating metrics for ${dataset_name} ---"
        
        if [ ! -d "$SR_OUTPUT_PATH" ]; then
            echo "ERROR: Super-Resolution directory not found at '$SR_OUTPUT_PATH'."
            echo "Cannot calculate metrics. Please run generation first or check the path."
            echo
            continue
        fi
        
        metric_cmd_args=(
            --sr_dir "$SR_OUTPUT_PATH"
            --dataset "$dataset_name"
            --name "${MODEL_NAME}"
        )

        if [ -d "$GT_PATH" ]; then
            echo "Found Ground Truth directory: $GT_PATH"
            metric_cmd_args+=(--gt_dir "$GT_PATH")
        else
            echo "No Ground Truth directory found for $dataset_name. Calculating No-Reference metrics only."
        fi

        python -W ignore "$METRIC_SCRIPT" "${metric_cmd_args[@]}"
        echo "--- Metrics for ${dataset_name} complete. ---"
        echo
    done
done

echo "######################################################################"
echo "## All checkpoints and datasets processed successfully!"
echo "######################################################################"