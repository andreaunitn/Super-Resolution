python -W ignore sam2_processing.py \
  --image_dir="preset/datasets/train_datasets/LSDIR/paired_data/gt" \
  --gt_seg_dir="preset/datasets/train_datasets/LSDIR/paired_data/gt_seg" \

python -W ignore sam2_processing.py \
  --image_dir="preset/datasets/train_datasets/LSDIR/sr_bicubic" \
  --embed_dir="preset/datasets/train_datasets/LSDIR/sam_embeds" \
  --seg_embed_dir="preset/datasets/train_datasets/LSDIR/seg_embeds" \