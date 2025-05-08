import os

import torch
from torchvision import transforms

import pyiqa
import numpy as np
from PIL import Image   

from basicsr.metrics.psnr_ssim import calculate_psnr_pt, calculate_ssim_pt

import wandb
wandb.login()

SR_PREFIX = "preset/datasets/output/"
GT_PREFIX = "preset/datasets/test_datasets/"

FULL_REFERENCE_METRICS = ["psnr", "ssim", "lpips", "dists"]
NO_REFERENCE_METRICS = ["niqe", "maniqa", "musiq", "clipiqa"]

def get_metrics(metrics, dataset):
    
    gt_dir = os.path.join(GT_PREFIX, dataset + "/test_HR")
    
    metrics_to_log = {}
    # metrics_to_log["dataset"] = dataset

    for metric in NO_REFERENCE_METRICS:
        metrics_to_log[metric] = np.mean(metrics[metric])
        print(f"{metric}: {metrics_to_log[metric]:.4f}")

    if not os.path.exists(gt_dir):
        return metrics_to_log

    for metric in FULL_REFERENCE_METRICS:
        metrics_to_log[metric] = np.mean(metrics[metric])
        print(f"{metric}: {metrics_to_log[metric]:.4f}")

    metrics_to_log["fid"] = metrics["fid"]
    print(f"fid: {metrics_to_log['fid']:.4f}")

    return metrics_to_log

def compute_metrics(args):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    img_preproc = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset_level_metrics = {}

    for dataset in args.datasets:

        sr_dir = os.path.join(SR_PREFIX, dataset + "/sample00")
        gt_dir = os.path.join(GT_PREFIX, dataset + "/test_HR")

        print("######################################################################################")
        print(f"Initializing IQA metric models for {dataset}")
        print("######################################################################################")

        metrics = {
            "niqe": [],
            "maniqa": [],
            "musiq": [],
            "clipiqa": [],
        }

        if os.path.exists(gt_dir):
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

        processed_count = 0
        image_files = sorted(os.listdir(sr_dir))
        num_images = len(image_files)

        print("######################################################################################")
        print(f"Calculating metrics for {dataset}")
        print("######################################################################################")
        for i, image in enumerate(image_files):
            sr_path = os.path.join(sr_dir, image)
            sr_image = img_preproc(Image.open(sr_path).convert("RGB")).unsqueeze(0).to(device)

            if os.path.exists(gt_dir):
                gt_path = os.path.join(gt_dir, image)
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
        
        if os.path.exists(gt_dir):
            # pyiqa compute FID by passing directly the gt and sr folders
            fid = fid_iqa_metric(gt_dir, sr_dir)
            metrics["fid"] = fid

        # log metrics for wandb
        metrics_to_log = get_metrics(metrics, dataset)
        dataset_level_metrics[dataset] = metrics_to_log

    with wandb.init(project="thesis", name="metric summary"):
        for metric in FULL_REFERENCE_METRICS + NO_REFERENCE_METRICS + ["fid"]:
            table = wandb.Table(columns=["dataset", metric])
            for dataset, metric_dict in dataset_level_metrics.items():
                if metric in metric_dict:
                    table.add_data(dataset, metric_dict[metric])

            wandb.log({f"{metric}_bar_chart": wandb.plot.bar(
                table,
                "dataset",
                metric,
                "dataset",
            )})

    wandb.finish()
