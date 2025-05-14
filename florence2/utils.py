import torch

import matplotlib.pyplot as plt  
import matplotlib.patches as patches

from torchvision.transforms.functional import to_pil_image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def preproc(pixel_values):

    tensor = pixel_values[0].cpu()

    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std = torch.tensor(IMAGENET_STD)[:, None, None]

    img_tensor = tensor * std + mean
    img_tensor = img_tensor.clamp(0, 1)
    image = to_pil_image(img_tensor)
    
    return image

"""
def denormalize_tensor(tensor, mean, std):

    # Ensure mean and std are tensors and correctly shaped for broadcasting
    mean_tensor = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std_tensor = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)

    denormalized = tensor.clone() # Operate on a clone
    denormalized.mul_(std_tensor).add_(mean_tensor) # tensor = tensor * std + mean

    return denormalized
"""
    
def plot_bbox(image, data):
    """
    Plots bounding boxes on an image.

    Parameters:
    - image: The image to plot on.
    - data: A dictionary containing 'bboxes' and 'labels'.
        - 'bboxes': A list of bounding boxes, where each box is a list of [x1, y1, x2, y2].
        - 'labels': A list of labels corresponding to each bounding box.
    """
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
      
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)  
      
    # Plot each bounding box  
    for bbox, label in zip(data['bboxes'], data['labels']):  
        x1, y1, x2, y2 = bbox  
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')    
        ax.add_patch(rect)  
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))  
      
    ax.axis('off')  
      
    plt.show()