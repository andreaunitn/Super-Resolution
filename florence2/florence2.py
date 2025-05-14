from transformers import AutoProcessor, AutoModelForCausalLM
from .utils import preproc
import torch

def eval_image(model, processor, task_prompt, image, text_input=None):

    # bboxes
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    inputs = processor(text=prompt, images=image, return_tensors="pt").to('cuda', torch.float16)
    pixel_values = inputs["pixel_values"].cuda()
    
    generated_ids = model.generate(
      input_ids=inputs["input_ids"].cuda(),
      pixel_values=pixel_values,
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )

    # embeds
    embeds = model._encode_image(pixel_values).mean(dim=1)[0]

    return parsed_answer, embeds

def load_florence2():
    
    model_id = "microsoft/Florence-2-base"
    florence = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype="auto").eval().cuda()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    return florence, processor

def extract_embeds_and_bboxes(pixel_values):

    florence2, processor = load_florence2()

    task_prompt = "<CAPTION>"
    image = preproc(pixel_values)
    bboxes, embeds = eval_image(florence2, processor, task_prompt, image)

    return bboxes, embeds
