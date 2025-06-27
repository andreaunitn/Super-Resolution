import os

import torch
from torchvision import transforms

import pyiqa
import numpy as np
from PIL import Image   

from basicsr.metrics.psnr_ssim import calculate_psnr_pt, calculate_ssim_pt

import wandb
wandb.login()

import json

def get_metrics(metrics):

    dataset_metrics = []

    for metric in list(metrics.keys()):
        dataset_metrics.append((metric, np.mean(metrics[metric])))

    return dataset_metrics

def compute_metrics(args):

    SR_PREFIX = "preset/datasets/output/"
    GT_PREFIX = "preset/datasets/test_datasets/"
    METRICS_PATH = "metrics/metrics.json"
    
    metrics_to_log = {}
    if not args.recompute_metrics and os.path.isfile(METRICS_PATH):

        print("######################################################################################")
        print(f"Loading metrics from {METRICS_PATH}")
        print("######################################################################################")

        with open(METRICS_PATH, "r") as f:
            metrics_to_log = json.load(f)
    else:

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        img_preproc = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        for dataset in args.datasets:

            sr_dir = os.path.join(SR_PREFIX, dataset + "/sample00")
            gt_dir = os.path.join(GT_PREFIX, dataset + "/test_HR_10_img")

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

            with torch.autocast("cuda", dtype=torch.float32):
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

            # metrics to log per dataset
            metrics_to_log[dataset] = get_metrics(metrics)

        # saving metrics
        with open(METRICS_PATH, "w") as f:
            json.dump(metrics_to_log, f, indent=4)

    HIGHER_IS_BETTER = ["psnr", "ssim", "maniqa", "musiq", "clipiqa"]

    for metric in metrics_to_log["DRealSR"]:
        print(f"{metric=}")

    # log metrics for wandb
    ALL_METRICS = sorted({
        m for metric_list in metrics_to_log.values()
        for m, _ in metric_list
    })

    for metric in ALL_METRICS:
        wandb.init(
            project="thesis",
            name=metric,
            id=metric + "-avg",
            reinit=True,
            # resume="allow",
            group="metric-comparison"
        )

        table = wandb.Table(columns=["dataset", "value"])

        for dataset, metric_list in metrics_to_log.items():
            metric_dict = dict(metric_list)
            if metric in metric_dict:
                table.add_data(dataset, metric_dict[metric])

        if len(table.data) > 0:
            chart = wandb.plot.bar(
                table,
                "dataset",
                "value",
                title=metric + "↑" if metric in HIGHER_IS_BETTER else metric + "↓"
            )

            wandb.log({f"{metric}_bar_chart": chart})

        wandb.finish()