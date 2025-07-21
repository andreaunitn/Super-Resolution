python -W ignore utils_data/sam2_processing.py \
  --image_dir="preset/datasets/train_datasets/LSDIR/gt" \
  --mask_dir="preset/datasets/train_datasets/LSDIR/gt_seg"

python -W ignore utils_data/sam2_processing.py \
  --image_dir="preset/datasets/train_datasets/LSDIR/sr_bicubic" \
  --embed_dir="preset/datasets/train_datasets/LSDIR/sam_embeds" \
  --logit_dir="preset/datasets/train_datasets/LSDIR/seg_embeds" \