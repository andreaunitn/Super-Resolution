import torch
import torch.nn.functional as F

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

import argparse
import cv2
import os
from tqdm import tqdm

def load():

    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    sam2_model = SAM2AutomaticMaskGenerator.from_pretrained("facebook/sam2.1-hiera-tiny", 
                                                                apply_postprocessing=False,
                                                                points_per_batch=128,
                                                                stability_score_thresh=0.5,
                                                                )

    return sam2_model

def pad_existing_mask_tensor(mask_tensor, max_n, device):
    
    current_n = mask_tensor.shape[0]
    
    if current_n > max_n:
        return mask_tensor[:max_n]
    
    elif current_n < max_n:

        pad_size = max_n - current_n
        padding = torch.zeros(pad_size, 512, 512, device=device, dtype=torch.float)
        return torch.cat([mask_tensor, padding], dim=0)
    else:

        return mask_tensor

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default=None, help="Path of the directory of the GT images")
    parser.add_argument("--gt_seg_dir", type=str, default=None, help="Path to the directory of the segmentation mask images")
    parser.add_argument("--max_seg", type=int, default=500, help="Number of masks to keep per image")
    args = parser.parse_args()

    sam2_model = load()

    if not os.path.exists(args.gt_seg_dir):
        os.makedirs(args.gt_seg_dir)

    images = sorted(os.listdir(args.image_dir))

    with torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.inference_mode():
            with tqdm(total=len(images), desc="Progress", unit="img") as pbar:
                for image in images:

                    img = Image.open(os.path.join(args.image_dir, image)).convert("RGB")
                    img = np.array(img)
                    
                    masks = sam2_model.generate(img)
                    masks = sorted(masks, key=lambda x: x['area'], reverse=True)
                    masks = masks[:args.max_seg]

                    masks_tensor = torch.stack([torch.from_numpy(m['segmentation']) for m in masks]).to("cuda").float()

                    pbar.set_postfix({
                        "file"     : image,
                        "num_masks": len(masks),
                    })

                    torch.save(masks_tensor, os.path.join(args.gt_seg_dir, image.replace(".png", ".pt")))

                    pbar.update(1)
