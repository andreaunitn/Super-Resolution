import os

import torch
from torchvision import transforms

import pyiqa
import numpy as np
from PIL import Image   

from basicsr.metrics.psnr_ssim import calculate_psnr_pt, calculate_ssim_pt

import wandb
wandb.login()

def compute_metrics(args):

    run = wandb.init(
        project="thesis",
        name="metrics",
    )

    sr_dir = os.path.join(args.output_dir, "sample00")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    img_preproc = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    print("######################################################################################")
    print("Initializing IQA metric models")
    print("######################################################################################")
    psnr_values = []
    ssim_values = []
    lpips_values = []
    lpips_iqa_metric = pyiqa.create_metric('lpips', device=device)
    dists_values = []
    dists_iqa_metric = pyiqa.create_metric('dists', device=device)
    fid_iqa_metric = pyiqa.create_metric('fid', device=device)
    niqe_values = []
    niqe_iqa_metric = pyiqa.create_metric('niqe', device=device)
    maniqa_values = []
    maniqa_iqa_metric = pyiqa.create_metric('maniqa', device=device)
    musiq_values = []
    musiq_iqa_metric = pyiqa.create_metric('musiq', device=device)
    clip_values = []
    clip_iqa_metric = pyiqa.create_metric('clipiqa', device=device)

    processed_count = 0
    image_files = sorted(os.listdir(args.gt_dir))
    num_images = len(image_files)

    print("######################################################################################")
    print("Calculating metrics")
    print("######################################################################################")
    for i, image in enumerate(image_files):
        gt_path = os.path.join(args.gt_dir, image)
        sr_path = os.path.join(sr_dir, image)

        gt_image = img_preproc(Image.open(gt_path).convert("RGB")).unsqueeze(0).to(device)
        sr_image = img_preproc(Image.open(sr_path).convert("RGB")).unsqueeze(0).to(device)

        psnr = calculate_psnr_pt(gt_image, sr_image, crop_border=4, test_y_channel=True).item()
        ssim = calculate_ssim_pt(gt_image, sr_image, crop_border=4, test_y_channel=True).item()
        lpips = lpips_iqa_metric(sr_image, gt_image).item()
        dists = dists_iqa_metric(sr_image, gt_image).item()
        niqe = niqe_iqa_metric(sr_image).item()
        maniqa = maniqa_iqa_metric(sr_image, gt_image).item()
        musiq = musiq_iqa_metric(sr_image).item()
        clipiqa = clip_iqa_metric(sr_image).item()

        psnr_values.append(psnr)
        ssim_values.append(ssim)
        lpips_values.append(lpips)
        dists_values.append(dists)
        niqe_values.append(niqe)
        maniqa_values.append(maniqa)
        musiq_values.append(musiq)
        clip_values.append(clipiqa)

        processed_count += 1
        print(f"Image {i+1}/{num_images}: {image}")

    # pyiqa compute FID by passing directly the gt and sr folders
    fid = fid_iqa_metric(args.gt_dir, sr_dir)

    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_lpips = np.mean(lpips_values)
    avg_dists = np.mean(dists_values)
    avg_niqe = np.mean(niqe_values)
    avg_maniqa = np.mean(maniqa_values)
    avg_musiq = np.mean(musiq_values)
    avg_clip = np.mean(clip_values)

    print(f"PSNR ↑: {avg_psnr:.2f}")
    print(f"SSIM ↑: {avg_ssim:.4f}")
    print(f"LPIPS ↓: {avg_lpips:.4f}")
    print(f"DISTS ↓: {avg_dists:.4f}")
    print(f"FID ↓: {fid:.2f}")
    print(f"NIQE ↓: {avg_niqe:.4f}")
    print(f"MANIQA ↑: {avg_maniqa:.4f}")
    print(f"MUSIQ ↑: {avg_musiq:.4f}")
    print(f"CLIPIQA ↑: {avg_clip:.4f}")

    # log metrics to wandb
    metrics_to_log = {
        "PSNR": avg_psnr,
        "SSIM": avg_ssim,
        "LPIPS": avg_lpips,
        "DISTS": avg_dists,
        "FID": fid,
        "NIQE": avg_niqe,
        "MANIQA": avg_maniqa,
        "MUSIQ": avg_musiq,
        "CLIPIQA": avg_clip
    }

    run.log(metrics_to_log)
    run.finish()
