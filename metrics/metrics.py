import os

import torch
from torchvision import transforms

import numpy as np

from PIL import Image

from basicsr.metrics.psnr_ssim import calculate_psnr_pt, calculate_ssim_pt

def calculate_psnr(args):
    """
    Calculate PSNR for the images in the specified directory.
    
    Args:
        args: Command line arguments containing the output directory and ground truth directory.
    """

    sr_dir = os.path.join(args.output_dir, "sample00")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    img_preproc = transforms.Compose([
        transforms.ToTensor(),
    ])

    psnr_values = []
    for image in os.listdir(args.gt_dir):
        gt_path = os.path.join(args.gt_dir, image)
        sr_path = os.path.join(sr_dir, image)

        gt_image = img_preproc(Image.open(gt_path).convert("RGB")).unsqueeze(0).to(device)
        sr_image = img_preproc(Image.open(sr_path).convert("RGB")).unsqueeze(0).to(device)

        psnr = calculate_psnr_pt(gt_image, sr_image, crop_border=4, test_y_channel=True).item()
        psnr_values.append(psnr)

    return np.mean(psnr_values)

def calculate_ssim(args):
    """
    Calculate SSIM for the images in the specified directory.
    
    Args:
        args: Command line arguments containing the output directory and ground truth directory.
    """

    sr_dir = os.path.join(args.output_dir, "sample00")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    img_preproc = transforms.Compose([
        transforms.ToTensor(),
    ])

    ssim_values = []
    for image in os.listdir(args.gt_dir):
        gt_path = os.path.join(args.gt_dir, image)
        sr_path = os.path.join(sr_dir, image)

        gt_image = img_preproc(Image.open(gt_path).convert("RGB")).unsqueeze(0).to(device)
        sr_image = img_preproc(Image.open(sr_path).convert("RGB")).unsqueeze(0).to(device)

        ssim = calculate_ssim_pt(gt_image, sr_image, crop_border=4, test_y_channel=True).item()
        ssim_values.append(ssim)

    return np.mean(ssim_values)

def compute_metrics(args): 
    psnr = calculate_psnr(args)
    ssim = calculate_ssim(args)
    
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")
