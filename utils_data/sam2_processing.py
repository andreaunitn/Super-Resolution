# import torch
# import torch.nn.functional as F

# import numpy as np
# from PIL import Image

# import matplotlib.pyplot as plt

# import argparse
# import cv2
# import os
# from tqdm import tqdm

# from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# def load(model_size="tiny", points_per_batch=None, points_per_side=None, stability_score_thresh=0.5, apply_postprocessing=False):

#     model_name_map = {
#         "tiny": "facebook/sam2.1-hiera-tiny",
#         "large": "facebook/sam2.1-hiera-large"
#     }

#     if model_size not in model_name_map:
#         raise ValueError(f"Invalid model size: {model_size}. Choose 'tiny' or 'large'.")

#     print(f"\nLoading SAM2 model: {model_name_map[model_size]}")

#     generator_params = {
#         "apply_postprocessing": apply_postprocessing,
#         "stability_score_thresh": stability_score_thresh,
#     }

#     if points_per_batch is not None:
#         generator_params["points_per_batch"] = points_per_batch

#     if points_per_side is not None:
#         generator_params["points_per_side"] = points_per_side

#     torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
#     if torch.cuda.get_device_properties(0).major >= 8:
#         torch.backends.cuda.matmul.allow_tf32 = True
#         torch.backends.cudnn.allow_tf32 = True

#     sam2_model = SAM2AutomaticMaskGenerator.from_pretrained(model_name_map[model_size], **generator_params)
#     print("SAM2 model loaded successfully.\n")

#     return sam2_model

# def get_seg_embeds(masks, image_embed):
#     """
#     Calculates per-mask embeddings from the full image embedding.
#     """
#     C = image_embed.shape[1]
#     if not masks:
#         return torch.zeros((0, C), device=image_embed.device)

#     per_mask_embeds = []
#     for ann in masks:
#         # Create a mask tensor and downsample it
#         seg = torch.from_numpy(ann["segmentation"][None, None].astype(np.float32))
#         seg_small = F.interpolate(seg, size=image_embed.shape[-2:], mode="nearest").to(image_embed.device)
        
#         # Apply mask and average-pool
#         masked = image_embed * seg_small
#         sum_mask = seg_small.sum()
#         if sum_mask > 1e-6:
#             masked_sum_per_channel = (image_embed * seg_small).sum(dim=(2, 3))
#             embed_vec = masked_sum_per_channel / sum_mask
#         else:
#             embed_vec = torch.zeros((1, C), device=image_embed.device)

#         per_mask_embeds.append(embed_vec.squeeze(0))

#     return torch.stack(per_mask_embeds, dim=0) if per_mask_embeds else torch.zeros((0, C), device=image_embed.device)

# def pad_existing_mask_tensor(mask_tensor, max_n, device):
#     """
#     Pads or truncates a tensor of masks to a fixed number (max_n).
#     """
#     current_n = mask_tensor.shape[0]
#     if current_n == 0:
#         return torch.zeros(max_n, 512, 512, device=device, dtype=torch.float)

#     h, w = mask_tensor.shape[-2:]
    
#     if current_n > max_n:
#         return mask_tensor[:max_n]
#     elif current_n < max_n:
#         pad_size = max_n - current_n
#         padding = torch.zeros(pad_size, h, w, device=device, dtype=torch.float)
#         return torch.cat([mask_tensor, padding], dim=0)
#     else:
#         return mask_tensor

# def show_anns(img, anns, borders=False, save_path=None):
#     """
#     Visualizes the generated masks on top of an image.
#     Optionally saves the figure to a file.
#     """
#     if not anns:
#         print("Warning: No annotations to show.")
#         return

#     anns = sorted(anns, key=lambda x: x['area'], reverse=True)
#     h, w = anns[0]['segmentation'].shape
#     mask_canvas = np.zeros((h, w, 4), dtype=float)

#     for ann in anns:
#         m = ann['segmentation'].astype(bool)
#         color = np.concatenate([np.random.rand(3), [0.5]]) # RGBA with alpha
#         mask_canvas[m] = color

#         if borders:
#             contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#             cv2.drawContours(mask_canvas, contours, -1, (0, 0, 1, 0.8), thickness=2)

#     fig, ax = plt.subplots(figsize=(10, 10))
#     ax.imshow(img)    
#     ax.imshow(mask_canvas)
#     ax.axis('off')
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         print(f"Saved visualization to {save_path}")
#         plt.close(fig)
#     else:
#         plt.show()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Unified SAM2 processing script to generate masks and embeddings.")
    
#     # --- I/O Arguments ---
#     parser.add_argument("--image_dir", type=str, required=True, help="Path to the directory of source images.")
#     parser.add_argument("--gt_seg_dir", type=str, default=None, help="Path to save the ground-truth segmentation mask tensors (.pt files).")
#     parser.add_argument("--embed_dir", type=str, help="Path to save the full image embeddings (.pt files).")
#     parser.add_argument("--seg_embed_dir", type=str, help="Path to save the per-mask segmentation embeddings (.pt files).")
#     parser.add_argument("--vis_dir", type=str, help="Optional: Path to save visualization images of the masks.")

#     # --- Model and Generation Arguments ---
#     parser.add_argument("--max_seg", type=int, default=150, help="Maximum number of masks to keep per image, sorted by area.")
#     parser.add_argument("--points_per_side", type=int, help="Points per side for mask generation (for 'large' model).")
#     parser.add_argument("--points_per_batch", type=int, default=128, help="Points processed in a batch for mask generation.")
#     parser.add_argument("--stability_score_thresh", type=float, default=0.5, help="Stability score threshold for filtering masks.")
    
#     args = parser.parse_args()

#     # Determine which tasks to run based on provided directories
#     do_gt_seg = args.gt_seg_dir is not None
#     do_embeds = args.embed_dir is not None and args.seg_embed_dir is not None
#     do_vis = args.vis_dir is not None

#     if not (do_gt_seg or do_embeds or do_vis):
#         print("Error: You must specify at least one output directory (--gt_seg_dir, --embed_dir, or --vis_dir).")
#         exit()

#     # Create directories
#     if do_gt_seg: os.makedirs(args.gt_seg_dir, exist_ok=True)
#     if do_embeds:
#         os.makedirs(args.embed_dir, exist_ok=True)
#         os.makedirs(args.seg_embed_dir, exist_ok=True)
#     if do_vis: os.makedirs(args.vis_dir, exist_ok=True)

#     # Load model
#     if do_gt_seg:
#         sam2_model = load(
#             model_size="tiny",
#             apply_postprocessing=False,
#             stability_score_thresh=0.5,
#         )
        
#     else:
#         sam2_model = load(
#             model_size="large",
#             apply_postprocessing=False,
#             points_per_side=16,
#             points_per_batch=512,
#             stability_score_thresh=0.0,
#         )
    
#     image_files = sorted([f for f in os.listdir(args.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

#     with torch.inference_mode():
#         for image_name in tqdm(image_files, desc="Processing images"):
#             base_name = os.path.splitext(image_name)[0]
#             image_path = os.path.join(args.image_dir, image_name)

#             img_np = np.array(Image.open(image_path).convert("RGB"))

#             masks = sam2_model.generate(img_np)

#             masks = sorted(masks, key=lambda x: x['area'], reverse=True)
#             if len(masks) > args.max_seg:
#                 masks = masks[:args.max_seg]

#             # Save Ground-Truth Segmentation Masks
#             if do_gt_seg:
#                 gt_masks_tensor = torch.stack([torch.from_numpy(m['segmentation']) for m in masks])
#                 torch.save(gt_masks_tensor.cpu(), os.path.join(args.gt_seg_dir, f"{base_name}.pt"))

#             # Save Image and Segmentation Embeddings
#             if do_embeds:
#                 sam2_model.predictor.set_image(img_np)
#                 image_embedding = sam2_model.predictor.get_image_embedding()
                
#                 torch.save(image_embedding.cpu(), os.path.join(args.embed_dir, f"{base_name}.pt"))
                
#                 seg_embeddings = get_seg_embeds(masks, image_embedding)
#                 torch.save(seg_embeddings.cpu(), os.path.join(args.seg_embed_dir, f"{base_name}.pt"))


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

def load(model_size="large", **kwargs):

    model_name_map = {"tiny": "facebook/sam2-hiera-tiny", "large": "facebook/sam2-hiera-large"}
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