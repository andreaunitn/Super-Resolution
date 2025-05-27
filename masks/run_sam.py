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

    # Reduce points_per_batch if it uses too much memory
    sam2_model = SAM2AutomaticMaskGenerator.from_pretrained("facebook/sam2.1-hiera-large", 
                                                            apply_postprocessing=False,
                                                            points_per_side=16,
                                                            points_per_batch=512,
                                                            stability_score_thresh=0.0,
                                                            )
    
    return sam2_model

def show_anns(img, anns, borders=False, title=None):

    if not anns:
        return

    anns = sorted(anns, key=lambda x: x['area'], reverse=True)
    h, w = anns[0]['segmentation'].shape

    mask_canvas = np.zeros((h, w, 4), dtype=float)

    for ann in anns:
        m = ann['segmentation'].astype(bool)
        color = np.concatenate([np.random.rand(3), [0.5]])
        mask_canvas[m] = color

        if borders:
            contours, _ = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            contours = [
                cv2.approxPolyDP(cnt, epsilon=0.01, closed=True)
                for cnt in contours
            ]
            cv2.drawContours(
                mask_canvas, contours, -1,
                (0, 0, 1, 0.8),
                thickness=1
            )

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)    
    ax.imshow(mask_canvas)
    ax.axis('off')
    plt.tight_layout()
    # plt.savefig(f"{title}", dpi=300, bbox_inches='tight')
    plt.show()

def get_seg_fig(anns):
    
    if not anns:
        return

    h, w = anns[0]['segmentation'].shape
    mask_canvas = np.zeros((h, w, 4), dtype=float)
    
    for ann in anns:
        m = ann['segmentation'].astype(bool)
        color = np.concatenate([np.random.rand(3), [0.5]])
        mask_canvas[m] = color

    return mask_canvas

def get_seg_embeds(masks, image_embed):

    # Return a tensor with 0s if no masks are present
    C = image_embed.shape[1]
    if len(masks) == 0:
        return torch.zeros((0, C), device=image_embed.device)

    per_mask_embeds = []

    for ann in masks:
        seg = torch.from_numpy(ann["segmentation"][None,None].astype(np.float32))
        
        # Down-sample the mask to match the spatial size of image_embed
        seg_small = F.interpolate(seg, size=image_embed.shape[-2:], mode="nearest").to(image_embed.device)

        # Apply the mask
        masked = image_embed * seg_small

        # Average-pool over the masked region
        sum_mask  = seg_small.sum(dim=[2,3])
        embed_vec = masked.sum(dim=[2,3]) / (sum_mask)

        per_mask_embeds.append(embed_vec.squeeze(0))

    # Stack the embeddings for each mask
    return torch.stack(per_mask_embeds, dim=0)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default=None, help="Path of the directory of the SR bicubic upsampled images")
    parser.add_argument("--embed_dir", type=str, default=None, help="Path to the directory of the image embeddings")
    parser.add_argument("--seg_dir", type=str, default=None, help="Path to the directory of the segmentation masks embeddings")
    parser.add_argument("--seg_fig_dir", type=str, default=None, help="Path to the directory of the segmentation mask images")
    parser.add_argument("--max_seg", type=int, default=143, help="Number of masks to keep per image")
    args = parser.parse_args()

    sam2_model = load()

    if not os.path.exists(args.embed_dir):
        os.makedirs(args.embed_dir)

    if not os.path.exists(args.seg_dir):
        os.makedirs(args.seg_dir)

    if not os.path.exists(args.seg_fig_dir):
        os.makedirs(args.seg_fig_dir)

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

                    seg_fig = get_seg_fig(masks)

                    img = sam2_model.predictor.set_image(img)
                    embeds = sam2_model.predictor.get_image_embedding()

                    seg_embeds = get_seg_embeds(masks, embeds)

                    pbar.set_postfix({
                        "file"     : image,
                        "num_masks": len(masks),
                    })

                    torch.save(embeds, os.path.join(args.embed_dir, image.replace(".png", ".pt")))
                    torch.save(seg_embeds, os.path.join(args.seg_dir, image.replace(".png", ".pt")))
                    torch.save(seg_fig, os.path.join(args.seg_fig_dir, image.replace(".png", ".pt")))

                    pbar.update(1)
