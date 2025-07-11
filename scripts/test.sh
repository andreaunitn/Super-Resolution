#!/bin/bash

# datasets=("DIV2K" "DRealSR" "RealLR200" "RealSR")
datasets=("RealSR")

GENERATION_SCRIPT="test_seesr.py"
METRIC_SCRIPT="utils_data/metrics.py"
MODEL_NAME="Finetuned_model"
SKIP_GENERATION=false

# Paths
PRETRAINED_MODEL_PATH="preset/models/stable-diffusion-2-base"
FINETUNED_MODEL_PATH="preset/train_output/LSDIR/checkpoint-2000"
RAM_FT_PATH="preset/models/DAPE.pth"

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
for dataset_name in "${datasets[@]}"; do
    echo "######################################################################"
    echo "## PROCESSING DATASET: $dataset_name"
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

echo "######################################################################"
echo "## All datasets processed successfully!"
echo "######################################################################"