python -W ignore utils_data/sam2_processing.py \
  --image_dir="preset/datasets/train_datasets/FFHQ/gt" \
  --mask_dir="preset/datasets/train_datasets/FFHQ/gt_seg"

python -W ignore utils_data/sam2_processing.py \
  --image_dir="preset/datasets/train_datasets/FFHQ/sr_bicubic" \
  --embed_dir="preset/datasets/train_datasets/FFHQ/sam_embeds" \
  --logit_dir="preset/datasets/train_datasets/FFHQ/seg_embeds" \