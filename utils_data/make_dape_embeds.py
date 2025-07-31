import sys
sys.path.append("ram/models")

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
dape_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default=None, help="Path of the directory of the SR bicubic upsampled images")
    parser.add_argument("--embed_dir", type=str, default=None, help="Path to the directory of the image embeddings")
    parser.add_argument("--ram_ft_path", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.embed_dir, exist_ok=True)
    
    DAPE = ram(pretrained='preset/models/ram_swin_large_14m.pth',
          pretrained_condition=args.ram_ft_path, 
          image_size=384,
          vit='swin_l')

    DAPE.eval().to("cuda")

    bicubic_paths = glob.glob(os.path.join(args.image_dir, "*.png"))
    for bicubic_path in bicubic_paths:
        img = Image.open(bicubic_path).convert("RGB")
        img = img_preproc(img)
        img = img.squeeze()
        
        dape_values = F.interpolate(img.unsqueeze(0), size=(384, 384), mode="bicubic")
        dape_values = dape_values.clamp(0.0, 1.0)
        dape_values = dape_normalize(dape_values).to("cuda")

        with torch.no_grad():
            dape_image_embeddings = DAPE.generate_image_embeds(dape_values)

        filename = os.path.basename(bicubic_path)
        filename = filename.replace(".png", ".pt")
        output_path = os.path.join(args.embed_dir, filename)

        torch.save(dape_image_embeddings, os.path.join(args.embed_dir, filename))