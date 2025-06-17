from ram.models.ram_lora import ram

from PIL import Image
import argparse
import glob
import os

import torch
import torch.nn.functional as F
from torchvision import transforms

# Preprocessing transforms
img_preproc = transforms.ToTensor()
img_resize = transforms.Resize((384, 384))
dape_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=None, help="Path to the directory of the dataset")
    parser.add_argument("--ram_ft_path", type=str, default=None)
    args = parser.parse_args()

    bicubic_dir = os.path.join(args.root_dir, 'sr_bicubic')
    gt_dir = os.path.join(args.root_dir, 'gt')
    tag_path = os.path.join(args.root_dir, 'tag')
    embed_dir = os.path.join(args.root_dir, 'dape_embeds')
    
    DAPE = ram(pretrained='preset/models/ram_swin_large_14m.pth',
          pretrained_condition=args.ram_ft_path, 
          image_size=384,
          vit='swin_l')

    DAPE.eval().to("cuda")

    # Embeds
    bicubic_paths = glob.glob(os.path.join(bicubic_dir, "*.png"))
    for bicubic_path in bicubic_paths:
        # Embeds
        sr_img = Image.open(bicubic_path).convert("RGB")
        sr_img = img_preproc(sr_img)
        sr_img = sr_img.squeeze()
        
        dape_values = F.interpolate(sr_img.unsqueeze(0), size=(384, 384), mode="bicubic")
        dape_values = dape_values.clamp(0.0, 1.0)
        dape_values = dape_normalize(dape_values).to("cuda")

        with torch.no_grad():
            dape_image_embeddings = DAPE.generate_image_embeds(dape_values)

        filename = os.path.basename(bicubic_path)
        filename = filename.replace(".png", ".pt")
        output_path = os.path.join(embed_dir, filename)

        torch.save(dape_image_embeddings, os.path.join(embed_dir, filename))

    # Tag
    gt_paths = glob.glob(os.path.join(args.gt_dir, "*.png"))
    for gt_path in gt_paths:
        gt_img = Image.open(gt_path).convert("RGB")
        gt_img = img_preproc(gt_img)
        gt_img = img_resize(gt_img)
        gt_img = dape_normalize(gt_img).unsqueeze(0).to("cuda")

        basename = os.path.basename(gt_path).split('.')[0]

        gt_captions = DAPE.generate_tag(gt_img)[0]
        gt_prompt = f"{gt_captions[0]},"
        tag_save_path = tag_path + f'/{basename}.txt'
        f = open(f"{tag_save_path}", "w")
        f.write(gt_prompt)
        f.close()
        print(f'The GT tag of {basename}.txt: {gt_prompt}')

