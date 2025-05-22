import glob
import os
from PIL import Image
import random

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

        root_folders = root_folders.split(',')
        for root_folder in root_folders:
            lr_path = root_folder + '/sr_bicubic'
            tag_path = root_folder + '/tag'
            gt_path = root_folder + '/gt'

            self.lr_list += glob.glob(os.path.join(lr_path, '*.png'))
            self.gt_list += glob.glob(os.path.join(gt_path, '*.png'))
            self.tag_path_list += glob.glob(os.path.join(tag_path, '*.txt'))

        assert len(self.lr_list) == len(self.gt_list)
        assert len(self.lr_list) == len(self.tag_path_list)

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
    
    """
    def visualize_image(self, image, title=None, boxes=None, seg=None):
        img = image.detach().cpu()
        img = to_pil_image(img)

        if not boxes and not seg:
            img.save(f"../Super-Resolution/figs/{title}.png")

        else:
            fig, ax = plt.subplots()
            ax.imshow(img)

            if boxes:
                for bbox, label in zip(boxes['polygons'], boxes['labels']):  
                    x1, y1, x2, y2 = bbox  
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')    
                    ax.add_patch(rect)  
                    plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))

            if seg:
                for region, data in seg.items():
                    polys = data.get('polygons', [])
                    labels = data.get('labels', [])
                    for poly_pts, label in zip(polys, labels):
                        # Convert flat list to Nx2 array
                        verts = np.array(poly_pts).reshape(-1, 2)
                        patch = patches.Polygon(verts,
                                                closed=True,
                                                linewidth=1,
                                                edgecolor='g',
                                                facecolor='g',
                                                alpha=0.4)
                        ax.add_patch(patch)
                        # Put label at first vertex
                        x0, y0 = verts[0]
                        ax.text(x0,
                                y0,
                                label,
                                color='white',
                                fontsize=8,
                                bbox=dict(facecolor='green', alpha=0.5))

            ax.axis('off')
            plt.tight_layout()
            plt.savefig(f"../Super-Resolution/figs/{title}.png")
    """

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

        lq_img = lq_img.squeeze()

        ram_values = F.interpolate(lq_img.unsqueeze(0), size=(384, 384), mode='bicubic')
        ram_values = ram_values.clamp(0.0, 1.0)
        example["ram_values"] = self.ram_normalize(ram_values.squeeze(0))

        """
        # SAM
        sam_img_embeds = get_img_embeds(self.sam, lq_img)
        example["sam_img_emdeds"] = sam_img_embeds
        print(f"{sam_img_embeds.shape=}")

        sam_masks = get_sam_masks(self.sam, lq_img)
        #print(f"{sam_masks=}")
        
        if self.img_num < self.num_images_to_save:

            #self.visualize_image(gt_img, title=f"gt_img{self.img_num}")
            #self.visualize_image(lq_img, title=f"lr_img_upscaled{self.img_num}")
            # print(f"{self.img_num=}")
            #print(result)
            #self.visualize_image(lq_img, title=f"lr_img_with_boxes{self.img_num}", seg=result)
            visualize_image_masks(Image.open(lq_path).convert('RGB'), sam_masks, title=f"lr_img_with_seg_{self.img_num}")
            
            self.img_num += 1

        else:
            exit()
        """
        # example["florence_values"] = ram_values

        # plot_bbox(gt_img, title="gt_img")
        # plot_bbox(lq_img, title="lr_img")
        # plot_bbox(ram_values.squeeze(0), title="ram")

        # lq_img = lq_img.cuda()
        # model_id = "microsoft/Florence-2-base"
        # florence = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype="auto").eval().to(lq_img.device)
        # processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        # print(f"{lq_img.device=}")
        # image, bbox, florence_encoder_hidden_states = extract_embeds_and_bboxes(florence, processor, lq_img)
        # plot_bbox(image, boxes=bbox, title="bbox")

        # exit()

        return example

    def __len__(self):
        return len(self.gt_list)