import glob
import os
from PIL import Image
import random

import torch
from torchvision import transforms
from torch.utils import data as data
import torch.nn.functional as F

class PairedCaptionDataset(data.Dataset):
    def __init__(
            self,
            root_folders=None,
            tokenizer=None,
            null_text_ratio=0.5,
    ):
        super(PairedCaptionDataset, self).__init__()

        self.null_text_ratio = null_text_ratio
        self.lr_list = []
        self.gt_list = []
        self.tag_path_list = []
        self.sam2_img_embeds_list = []
        self.sam2_seg_emebds_list = []
        self.dape_img_embeds_list = []

        root_folders = root_folders.split(',')
        for root_folder in root_folders:
            lr_path = root_folder + '/sr_bicubic'
            tag_path = root_folder + '/tag'
            gt_path = root_folder + '/gt'
            sam2_img_embeds_path = root_folder + '/img_embeds'
            sam2_seg_embeds_path = root_folder + '/seg_embeds'
            dape_img_embeds_path = root_folder + '/dape_embeds'

            self.lr_list += glob.glob(os.path.join(lr_path, '*.png'))
            self.gt_list += glob.glob(os.path.join(gt_path, '*.png'))
            self.tag_path_list += glob.glob(os.path.join(tag_path, '*.txt'))
            self.sam2_img_embeds_list += glob.glob(os.path.join(sam2_img_embeds_path, '*.pt'))
            self.sam2_seg_emebds_list += glob.glob(os.path.join(sam2_seg_embeds_path, '*.pt'))
            self.dape_img_embeds_list += glob.glob(os.path.join(dape_img_embeds_path, '*.pt'))

        assert len(self.lr_list) == len(self.gt_list)
        assert len(self.lr_list) == len(self.tag_path_list)
        assert len(self.lr_list) == len(self.sam2_img_embeds_list)
        assert len(self.lr_list) == len(self.sam2_seg_emebds_list)
        assert len(self.lr_list) == len(self.dape_img_embeds_list)

        self.img_preproc = transforms.Compose([
            transforms.ToTensor(),
        ])

        ram_mean = [0.485, 0.456, 0.406]
        ram_std = [0.229, 0.224, 0.225]
        self.ram_normalize = transforms.Normalize(mean=ram_mean, std=ram_std)

        self.tokenizer = tokenizer

    def tokenize_caption(self, caption=""):
        inputs = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        return inputs.input_ids

    def __getitem__(self, index):

        gt_path = self.gt_list[index]
        gt_img = Image.open(gt_path).convert('RGB')
        gt_img = self.img_preproc(gt_img)
        
        lq_path = self.lr_list[index]
        lq_img = Image.open(lq_path).convert('RGB')
        lq_img = self.img_preproc(lq_img)

        if random.random() < self.null_text_ratio:
            tag = ''
        else:
            tag_path = self.tag_path_list[index]
            file = open(tag_path, 'r')
            tag = file.read()
            file.close()

        example = dict()
        example["conditioning_pixel_values"] = lq_img.squeeze(0)
        example["pixel_values"] = gt_img.squeeze(0) * 2.0 - 1.0
        example["input_ids"] = self.tokenize_caption(caption=tag).squeeze(0)

        example["ram_values"] = torch.load(self.dape_img_embeds_list[index]).squeeze(0)
        example["sam2_img_embeds"] = torch.load(self.sam2_img_embeds_list[index])
        example["sam2_seg_embeds"] = torch.load(self.sam2_seg_emebds_list[index])

        return example

    def __len__(self):
        return len(self.gt_list)