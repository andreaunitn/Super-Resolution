from transformers import AutoProcessor, AutoModelForCausalLM
from .utils import preproc
import torch

def eval_image(model, processor, task_prompt, image, text_input=None):

    # bboxes
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    # plot_bbox(image, title="image")
    inputs = processor(text=prompt, images=image, return_tensors="pt").to('cuda', torch.float16)
    pixel_values = inputs["pixel_values"]
    # plot_bbox(pixel_values, title="pixel_values")
    # print(f"{pixel_values.device=}")

    # with torch.no_grad():
    generated_ids = model.generate(
    input_ids=inputs["input_ids"].cuda(),
    pixel_values=pixel_values.cuda(),
    max_new_tokens=1024,
    early_stopping=False,
    do_sample=False,
    num_beams=3,
    )

    # image = to_pil_image(image)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )

    # embeds
    embeds = model._encode_image(pixel_values).mean(dim=1)[0]

    return parsed_answer, embeds

def load():
    
    model_id = "microsoft/Florence-2-base"
    florence = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype="auto")
    # processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, do_rescale=True, do_normalize=True)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, do_rescale=True, do_normalize=True)

    return florence, processor

def extract_embeds_and_result(model, processor, pixel_values):

    task_prompt = "<REGION_TO_SEGMENTATION>"
    # image = pixel_values
    image = preproc(pixel_values)
    result, embeds = eval_image(model, processor, task_prompt, image)
    # print(bboxes)
    return result, embeds
