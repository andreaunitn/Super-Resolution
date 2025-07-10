import argparse
import os

import pyiqa
from PIL import Image
import numpy as np

import torch
from torchvision import transforms

from basicsr.metrics.psnr_ssim import calculate_psnr_pt, calculate_ssim_pt

import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--sr_dir",
        type=str,
        default=None,
    )

    args = parser.parse_args()
    device = torch.device("cuda")

    img_preproc = transforms.Compose([
        transforms.ToTensor(),
    ])

    print("######################################################################################")
    print(f"Initializing metric models")
    print("######################################################################################")

    metrics_to_log = {}
    metrics = {
        "niqe": [],
        "maniqa": [],
        "musiq": [],
        "clipiqa": [],
    }

    if args.gt_dir is not None:
        metrics["psnr"] = []
        metrics["ssim"] = []
        metrics["lpips"] = []
        metrics["dists"] = []
        metrics["fid"] = 0

        lpips_iqa_metric = pyiqa.create_metric('lpips', device=device)
        dists_iqa_metric = pyiqa.create_metric('dists', device=device)
        fid_iqa_metric = pyiqa.create_metric('fid', device=device)
    
    niqe_iqa_metric = pyiqa.create_metric('niqe', device=device)
    maniqa_iqa_metric = pyiqa.create_metric('maniqa', device=device)
    musiq_iqa_metric = pyiqa.create_metric('musiq', device=device)
    clip_iqa_metric = pyiqa.create_metric('clipiqa', device=device)

    with torch.autocast("cuda", dtype=torch.float32):
        processed_count = 0
        image_files = sorted(os.listdir(args.sr_dir))
        num_images = len(image_files)

    print("######################################################################################")
    print(f"Calculating metrics")
    print("######################################################################################")

    for i, image in enumerate(image_files):
        sr_path = os.path.join(args.sr_dir, image)
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
        fid = fid_iqa_metric(args.gt_dir, args.sr_dir)
        metrics["fid"] = fid

    m = []
    for metric in list(metrics.keys()):
        m.append((metric, np.mean(metrics[metric])))

    metrics_to_log[args.dataset] = m
    
    path = "metrics/"
    with open(os.path.join(path, args.dataset + ".json"), "w") as f:
        json.dump(metrics_to_log, f, indent=4)
