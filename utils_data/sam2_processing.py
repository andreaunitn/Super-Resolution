import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import cv2
import os
from tqdm import tqdm

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

def get_mask_logits_from_anns(sam_generator, anns):
    """
    Calculates per-mask embeddings by extracting the MASK DECODER's raw logits.
    This is done by re-predicting each mask from its generation prompt.
    """
    if not anns:
        return torch.empty(0, 1, 256, 256)

    all_mask_logits = []
    predictor = sam_generator.predictor
    device = "cuda"

    for ann in anns:
        point_coords = ann['point_coords']
        point_labels = np.ones(len(point_coords))
        point_coords_torch = torch.tensor(point_coords, device=device).unsqueeze(0)
        point_labels_torch = torch.tensor(point_labels, device=device).unsqueeze(0)

        _, scores, logits = predictor.predict(
            point_coords=point_coords_torch,
            point_labels=point_labels_torch,
            multimask_output=True,
        )

        original_iou = ann['predicted_iou']
        
        scores_tensor = torch.from_numpy(scores).to(device)
        original_iou_tensor = torch.tensor(original_iou, device=device)
        
        best_idx = torch.argmin(torch.abs(scores_tensor.squeeze() - original_iou_tensor))
        best_logit_numpy_slice = logits[best_idx, :, :]
        best_logit = torch.from_numpy(best_logit_numpy_slice).unsqueeze(0)
        
        all_mask_logits.append(best_logit)

    all_mask_logits_on_device = [logit.to(device) for logit in all_mask_logits]
    
    if not all_mask_logits_on_device:
        return torch.empty(0, 1, 256, 256, device=device)
        
    return torch.stack(all_mask_logits_on_device, dim=0)

def load(model_size="tiny", **kwargs):

    model_name_map = {"tiny": "facebook/sam2.1-hiera-tiny", "large": "facebook/sam2.1-hiera-large"}
    if model_size not in model_name_map: 
        raise ValueError(f"Invalid model size: {model_size}.")
    
    print(f"\nLoading SAM2 model: {model_name_map[model_size]}")

    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    sam2_model = SAM2AutomaticMaskGenerator.from_pretrained(model_name_map[model_size], **kwargs)
    print("SAM2 model loaded successfully.\n")
    
    return sam2_model

def show_anns(img, anns, borders=False, save_path=None):

    if not anns: 
        return
    
    anns = sorted(anns, key=lambda x: x['area'], reverse=True)
    h, w = anns[0]['segmentation'].shape; mask_canvas = np.zeros((h, w, 4), dtype=float)

    for ann in anns:
        m = ann['segmentation'].astype(bool); color = np.concatenate([np.random.rand(3), [0.5]])
        mask_canvas[m] = color
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(mask_canvas, contours, -1, (0, 0, 1, 0.8), thickness=2)

    fig, ax = plt.subplots(figsize=(10, 10)); ax.imshow(img); ax.imshow(mask_canvas); ax.axis('off')
    plt.tight_layout()

    if save_path: 
        plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close(fig)
    else: plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified SAM2 processing script to generate masks and all embeddings.")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the directory of source images.")
    parser.add_argument("--mask_dir", type=str, help="Path to save the final binary segmentation masks (.pt files).")
    parser.add_argument("--embed_dir", type=str, help="Path to save the full image embeddings (.pt files).")
    parser.add_argument("--logit_dir", type=str, help="Path to save the per-mask decoder logit embeddings (.pt files).")
    parser.add_argument("--vis_dir", type=str, help="Optional: Path to save visualization images of the masks.")
    parser.add_argument("--model_size", type=str, default="large", choices=["tiny", "large"], help="Size of the SAM2 model to use.")
    parser.add_argument("--max_seg", type=int, default=150, help="Maximum number of masks to keep per image, sorted by area.")
    parser.add_argument("--points_per_side", type=int, default=16, help="Points per side for mask generation grid.")
    parser.add_argument("--points_per_batch", type=int, default=128, help="Points processed in a batch for mask generation.")
    parser.add_argument("--stability_score_thresh", type=float, default=0.9, help="Stability score threshold for filtering masks.")
    args = parser.parse_args()

    do_masks = args.mask_dir is not None
    do_image_embed = args.embed_dir is not None
    do_logits = args.logit_dir is not None
    do_vis = args.vis_dir is not None

    if not any([do_masks, do_image_embed, do_logits, do_vis]):
        print("Error: Must specify at least one output directory (--mask_dir, --embed_dir, --logit_dir, or --vis_dir).")
        exit()

    for path in [args.mask_dir, args.embed_dir, args.logit_dir, args.vis_dir]:
        if path: os.makedirs(path, exist_ok=True)

    sam2_model = load(
        model_size=args.model_size,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        stability_score_thresh=args.stability_score_thresh,
    )
    
    image_files = sorted([f for f in os.listdir(args.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    with torch.inference_mode():
        for image_name in tqdm(image_files, desc="Processing images"):
            base_name = os.path.splitext(image_name)[0]
            image_path = os.path.join(args.image_dir, image_name)
            img_np = np.array(Image.open(image_path).convert("RGB"))

            masks = sam2_model.generate(img_np)
            masks = sorted(masks, key=lambda x: x['area'], reverse=True)
            if len(masks) > args.max_seg:
                masks = masks[:args.max_seg]
        
            if do_image_embed:
                sam2_model.predictor.set_image(img_np)
                image_embedding = sam2_model.predictor.get_image_embedding()
                
                torch.save(image_embedding.cpu(), os.path.join(args.embed_dir, f"{base_name}.pt"))

            if do_logits:
                logit_embeddings = get_mask_logits_from_anns(sam2_model, masks)
                torch.save(logit_embeddings.cpu(), os.path.join(args.logit_dir, f"{base_name}.pt"))

            if do_masks:
                if not masks:
                    saved_masks_tensor = torch.empty(0, *img_np.shape[:2], dtype=torch.bool)
                else:
                    saved_masks_tensor = torch.stack([torch.from_numpy(m['segmentation']) for m in masks])
                torch.save(saved_masks_tensor.cpu(), os.path.join(args.mask_dir, f"{base_name}.pt"))

            if do_vis:
                vis_path = os.path.join(args.vis_dir, f"{base_name}_visualization.png")
                show_anns(img_np, masks, borders=True, save_path=vis_path)

    print("\nProcessing complete.")