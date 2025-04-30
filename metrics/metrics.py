import os

import torch
from torchvision import transforms

import pyiqa
import numpy as np
from PIL import Image        

from basicsr.metrics.psnr_ssim import calculate_psnr_pt, calculate_ssim_pt

def compute_metrics(args):

    sr_dir = os.path.join(args.output_dir, "sample00")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print(device)

    img_preproc = transforms.Compose([
        transforms.ToTensor(),
    ])

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

    for image in os.listdir(args.gt_dir):
        gt_path = os.path.join(args.gt_dir, image)
        sr_path = os.path.join(sr_dir, image)

        gt_image = img_preproc(Image.open(gt_path).convert("RGB")).unsqueeze(0).to(device)
        sr_image = img_preproc(Image.open(sr_path).convert("RGB")).unsqueeze(0).to(device)

        psnr = calculate_psnr_pt(gt_image, sr_image, crop_border=4, test_y_channel=True).item()
        ssim = calculate_ssim_pt(gt_image, sr_image, crop_border=4, test_y_channel=True).item()
        lpips = lpips_iqa_metric(sr_image, gt_image).item()
        dists = dists_iqa_metric(sr_image, gt_image).item()
        niqe = niqe_iqa_metric(sr_image)
        maniqa = maniqa_iqa_metric(sr_image, gt_image).item()
        musiq = musiq_iqa_metric(sr_image)

        psnr_values.append(psnr)
        ssim_values.append(ssim)
        lpips_values.append(lpips)
        dists_values.append(dists)
        niqe_values.append(niqe)
        maniqa_values.append(maniqa)
        musiq_values.append(musiq)

    # pyiqa compute FID by passing directly the gt and sr folders
    fid = fid_iqa_metric(args.gt_dir, sr_dir)
    
    print(f"PSNR: {np.mean(psnr_values):.2f}")
    print(f"SSIM: {np.mean(ssim_values):.4f}")
    print(f"LPIPS: {np.mean(lpips_values):.4f}")
    print(f"DISTS: {np.mean(dists_values):.4f}")
    print(f"FID: {np.mean(fid):.2f}")
    print(f"NIQE: {np.mean(niqe_values):.4f}")
    print(f"MANIQA: {np.mean(maniqa_values):.4f}")
    print(f"MUSIQ: {np.mean(musiq_values):.4f}")
