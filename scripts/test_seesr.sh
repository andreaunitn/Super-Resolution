python -W ignore test_seesr.py \
--pretrained_model_path preset/models/stable-diffusion-2-base \
--prompt None \
--seesr_model_path preset/models/seesr \
--ram_ft_path preset/models/DAPE.pth \
--image_path preset/datasets/test_datasets/DRealSR/test_LR \
--gt_dir preset/datasets/test_datasets/DRealSR/test_HR \
--output_dir preset/datasets/output/DRealSR \
--start_point lr \
--num_inference_steps 50 \
--guidance_scale 5.5 \
--process_size 512

