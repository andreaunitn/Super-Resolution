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
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoImageProcessor

# Preprocessing transforms
img_preproc = transforms.ToTensor()
img_resize = transforms.Resize((384, 384))
dape_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

def clean_and_format_tags(raw_text):
    """
    Cleans raw text from a tag file, filters invalid tags, deduplicates,
    and formats the result as a comma-separated string with a trailing comma.
    """

    tags = re.sub(r'[\n\s;.]+', ',', raw_text)
    tags = re.sub(r'[^a-zA-Z0-9, ]', '', tags)
    
    tag_list = [tag.strip().lower() for tag in tags.split(',') if tag.strip()]
    
    valid_tags = []
    for tag in tag_list:
        if tag.isnumeric():
            continue

        if len(tag.split()) > 2:
            continue
            
        valid_tags.append(tag)

    seen = set()
    unique_tags = []
    for tag in valid_tags:
        if tag not in seen:
            seen.add(tag)
            unique_tags.append(tag)

    if not unique_tags:
        return ""
    
    formatted_string = ", ".join(unique_tags) + ","
    
    return formatted_string

def process_directory(tags_dir):

    """
    Finds all .txt files in a directory, reads them, cleans the tags using all rules,
    and overwrites the files with the cleaned content.
    """

    if not os.path.isdir(tags_dir):
        print(f"Error: Directory not found at '{tags_dir}'")
        return

    txt_files = glob.glob(os.path.join(tags_dir, "*.txt"))

    if not txt_files:
        print(f"No .txt files found in '{tags_dir}'. Nothing to do.")
        return

    print(f"Found {len(txt_files)} existing files to clean in '{tags_dir}'.")
    
    processed_count = 0
    for file_path in txt_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()

            cleaned_content = clean_and_format_tags(original_content)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            
            processed_count += 1
        except Exception as e:
            print(f"Could not process file {file_path}: {e}")

    print(f"\nProcessing complete. Successfully cleaned and overwrote {processed_count} files.")

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=None, help="Path to the directory of the dataset")
    parser.add_argument("--ram_ft_path", type=str, default=None)
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf", help="Hugging Face model ID for LLaVA model.")
    parser.add_argument("--system_prompt", type=str,
                        default="You are a meticulous, expert image analyst. Your sole purpose is to describe images by generating a comprehensive, comma-separated list of tags. Be objective and precise.",
                        help="The high-level instruction for the model's role.")
    parser.add_argument(
                        "--user_prompt", 
                        type=str, 
                        default="""**Role:** You are a silent, highly-efficient image-to-tag converter.

                                **Task:** Analyze the image and generate a comma-separated list of tags based ONLY on its visual content.

                                **Rules:**
                                - Identify subjects, objects, the environment, and the overall style.
                                - Each tag must be a single word and unique.

                                **Format:**
                                - Start IMMEDIATELY with the first tag.
                                - Separate every tag with a single comma and a space.
                                - End the entire list with a final comma.
                                """,
                        help="The specific user request for the image."
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Number of images to process at once.")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of tokens to generate per image.")
    args = parser.parse_args()

    bicubic_dir = os.path.join(args.root_dir, 'sr_bicubic')
    gt_dir = os.path.join(args.root_dir, 'gt')
    tag_path = os.path.join(args.root_dir, 'llava_tag')
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

    del DAPE
    torch.cuda.empty_cache()

    print("Loading LLaVA model and processor components...")

    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    tokenizer = AutoProcessor.from_pretrained(args.model_id)
    vision_tower_id = "openai/clip-vit-large-patch14-336"
    image_processor = AutoImageProcessor.from_pretrained(vision_tower_id)

    print("Model and components loaded successfully.")

    gt_paths = glob.glob(os.path.join(gt_dir, "*.png"))

    for gt_path in gt_paths:
        print(f"Processing: {os.path.basename(gt_path)}")

        gt_img = Image.open(gt_path).convert("RGB")

        full_user_prompt = f"{args.system_prompt}\n\n{args.user_prompt}"
        
        prompt = f"USER: <image>\n{full_user_prompt}\nASSISTANT:"

        text_inputs = tokenizer(prompt, return_tensors="pt")
        image_inputs = image_processor([gt_img], return_tensors="pt")

        inputs = {**text_inputs, **image_inputs}
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generate_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)

        text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
        
        response = ""
        if "ASSISTANT:" in text:
            response = text.split("ASSISTANT:")[1].strip()
        
        basename = os.path.basename(gt_path).split('.')[0]
        tag_save_path = os.path.join(tag_path, f'{basename}.txt')

        with open(tag_save_path, "w") as f:
            f.write(response)
            
        print(f'-> Saved tags for {basename}.txt')

    process_directory(tag_path)

