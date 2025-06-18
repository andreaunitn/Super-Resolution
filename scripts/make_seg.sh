python -W ignore utils_data/sam2_processing.py \
  --image_dir="preset/datasets/train_datasets/LSDIR/gt" \
  --gt_seg_dir="preset/datasets/train_datasets/LSDIR/gt_seg" \

python -W ignore utils_data/sam2_processing.py \
  --image_dir="preset/datasets/train_datasets/LSDIR/sr_bicubic" \
  --embed_dir="preset/datasets/train_datasets/LSDIR/sam_embeds" \
  --seg_embed_dir="preset/datasets/train_datasets/LSDIR/seg_embeds" \