import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from ram.models.ram_lora import ram

from PIL import Image
import argparse
import glob
import os
import re

import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoProcessor, LlavaForConditionalGeneration

# Preprocessing transforms
img_preproc = transforms.ToTensor()
dape_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

def create_batches(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def clean_and_format_tags(raw_text):

    # Remove assistant's introductory phrases and the final </s> token
    text = re.sub(r'^\s*(ASSISTANT:)?\s*(here are the tags:|the tags for this image are:|the tags are:|sure, here are the tags:)?\s*', '', raw_text, flags=re.IGNORECASE)
    text = text.replace("</s>", "").strip()

    # Replace newlines, multiple spaces, and other separators with a comma
    tags = re.sub(r'[\n\s;.]+', ',', text)
    
    # Remove any character that is not a letter, number, space, or comma
    tags = re.sub(r'[^a-zA-Z0-9, ]', '', tags)
    
    # Split by comma, strip whitespace from each tag, and filter out any empty tags
    tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
    
    # Join them back together into the desired format
    if not tag_list:
        return "" # Return empty string if no valid tags were found
        
    formatted_string = ", ".join(tag_list) + ","
    
    return formatted_string

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=None, help="Path to the directory of the dataset")
    parser.add_argument("--ram_ft_path", type=str, default=None)
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-1.5-7b-hf", help="Hugging Face model ID for LLaVA model.")
    parser.add_argument("--system_prompt", type=str,
                        default="You are a meticulous, expert image analyst. Your sole purpose is to describe images by generating a comprehensive, comma-separated list of tags. Be objective and precise.",
                        help="The high-level instruction for the model's role.")
    parser.add_argument(
                        "--user_prompt", 
                        type=str, 
                        default="""Analyze this image and generate a comma-separated list of keywords and tags. Cover these categories:
                                    1.  **Main Subjects:** Identify the primary objects or people.
                                    2.  **Setting/Environment:** Describe the location (e.g., city street, forest, beach, inside a room).
                                    3.  **Composition:** Note the shot type (e.g., close-up, wide shot, portrait).
                                    4.  **Lighting & Atmosphere:** Describe the lighting and mood (e.g., sunny, overcast, night, dramatic lighting, serene, chaotic).
                                    5.  **Aesthetics & Style:** Identify the artistic style (e.g., photorealistic, illustration, vintage, black and white, vibrant colors).
                                    Do not use full sentences. End the entire list with a final comma.""",
                        help="The specific user request for the image."
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Number of images to process at once.")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of tokens to generate per image.")
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

    # DAPE Embeds
    bicubic_paths = glob.glob(os.path.join(bicubic_dir, "*.png"))
    for bicubic_path in bicubic_paths:
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

    del DAPE, dape_values
    torch.cuda.empty_cache()

    print("Loading LLaVA model and processor...")

    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    processor = AutoProcessor.from_pretrained(args.model_id)

    print("Model and processor loaded successfully.")

    # Llava tag
    gt_paths = glob.glob(os.path.join(gt_dir, "*.png"))
    for gt_path in gt_paths:
        gt_img = Image.open(gt_path).convert("RGB")

        # Prepare prompt
        conversation = [
            {"role": "system", "content": args.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": args.user_prompt},
                ],
            },
        ]

        input = processor.apply_chat_template(
            conversation,
            images=[gt_img],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt"
        ).to(model.device, torch.bfloat16)
    
        with torch.no_grad():
            generate_ids = model.generate(**input, max_new_tokens=args.max_new_tokens, do_sample=False)

        full_decoded_texts = processor.batch_decode(generate_ids, skip_special_tokens=False)
        for i, text in enumerate(full_decoded_texts):
            final_tags = clean_and_format_tags(text)

        print(f"{final_tags=}")
        exit()

        basename = os.path.basename(gt_path).split('.')[0]
        tag_save_path = tag_path + f'/{basename}.txt'
        f = open(f"{tag_save_path}", "w")
        f.write(final_tags)
        f.close()
        print(f'The GT tag of {basename}.txt: {final_tags}')

