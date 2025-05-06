import os

import torch
from torchvision import transforms

import pyiqa
import numpy as np
from PIL import Image   

from basicsr.metrics.psnr_ssim import calculate_psnr_pt, calculate_ssim_pt

import wandb
wandb.login()

FULL_REFERENCE_METRICS = ["psnr", "ssim", "lpips", "dists"]
NO_REFERENCE_METRICS = ["niqe", "maniqa", "musiq", "clipiqa"]

def get_metrics(args, metrics):
    
    metrics_to_log = {}
    for metric in NO_REFERENCE_METRICS:
        metrics_to_log[metric] = np.mean(metrics[metric])
        print(f"{metric}: {metrics_to_log[metric]:.4f}")

    if args.gt_dir is None:
        return metrics_to_log

    for metric in FULL_REFERENCE_METRICS:
        metrics_to_log[metric] = np.mean(metrics[metric])
        print(f"{metric}: {metrics_to_log[metric]:.4f}")

    metrics_to_log["fid"] = metrics["fid"]
    print(f"fid: {metrics_to_log['fid']:.4f}")

    return metrics_to_log

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

    metrics = {
        "psnr": [],
        "ssim": [],
        "lpips": [],
        "dists": [],
        "fid": 0,
        "niqe": [],
        "maniqa": [],
        "musiq": [],
        "clipiqa": []
    }

    if args.gt_dir is not None:
        lpips_iqa_metric = pyiqa.create_metric('lpips', device=device)
        dists_iqa_metric = pyiqa.create_metric('dists', device=device)
        fid_iqa_metric = pyiqa.create_metric('fid', device=device)

    niqe_iqa_metric = pyiqa.create_metric('niqe', device=device)
    maniqa_iqa_metric = pyiqa.create_metric('maniqa', device=device)
    musiq_iqa_metric = pyiqa.create_metric('musiq', device=device)
    clip_iqa_metric = pyiqa.create_metric('clipiqa', device=device)

    processed_count = 0
    image_files = sorted(os.listdir(sr_dir))
    num_images = len(image_files)

    print("######################################################################################")
    print("Calculating metrics")
    print("######################################################################################")
    for i, image in enumerate(image_files):
        sr_path = os.path.join(sr_dir, image)
        sr_image = img_preproc(Image.open(sr_path).convert("RGB")).unsqueeze(0).to(device)

        if args.gt_dir is not None:
            gt_path = os.path.join(args.gt_dir, image)
            gt_image = img_preproc(Image.open(gt_path).convert("RGB")).unsqueeze(0).to(device)

            psnr = calculate_psnr_pt(gt_image, sr_image, crop_border=4, test_y_channel=True).item()
            ssim = calculate_ssim_pt(gt_image, sr_image, crop_border=4, test_y_channel=True).item()
            lpips = lpips_iqa_metric(sr_image, gt_image).item()
            dists = dists_iqa_metric(sr_image, gt_image).item()

            metrics["psnr"].append(psnr)
            metrics["ssim"].append(ssim)
            metrics["lpips"].append(lpips)
            metrics["dists"].append(dists)

        niqe = niqe_iqa_metric(sr_image).item()
        maniqa = maniqa_iqa_metric(sr_image).item()
        musiq = musiq_iqa_metric(sr_image).item()
        clipiqa = clip_iqa_metric(sr_image).item()

        metrics["niqe"].append(niqe)
        metrics["maniqa"].append(maniqa)
        metrics["musiq"].append(musiq)
        metrics["clipiqa"].append(clipiqa)

        processed_count += 1
        print(f"Image {i+1}/{num_images}: {image}")
    
    if args.gt_dir is not None:
        # pyiqa compute FID by passing directly the gt and sr folders
        fid = fid_iqa_metric(args.gt_dir, sr_dir)
        metrics["fid"] = fid

    # log metrics for wandb
    metrics_to_log = get_metrics(args, metrics)
    run.log(metrics_to_log)
    run.finish()
