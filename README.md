<div align=center class="logo">
      <img src="figs/samsr.png" style="width:640px">
   </a>
</div>

# SamSR: Semantic-Aware Super-Resolution via SAM 2 Priors

**SamSR** is an advanced image super-resolution diffusion framework that leverages the powerful semantic capabilities of **Segment Anything Model 2 (SAM 2)**. Building upon the foundation of [SeeSR](https://github.com/cswry/SeeSR), this project introduces novel architectural improvements and optimization strategies to enhance generation quality and efficiency.

## Architecture Overview


## 🔎 Key Innovations

Unlike traditional super-resolution methods, SamSR integrates segmentation-aware priors to guide the diffusion process. Key contributions include:

*   **SAM 2-Guided Generation**: Introduced two new semantic priors derived from SAM 2:
    *   **SICA (SAM Image Cross-Attention)**: Leverages SAM 2 image embeddings from the [Hiera Encoder](https://huggingface.co/docs/transformers/model_doc/hiera).
    *   **SMCA (SAM Masks Cross-Attention)**: Utilizes SAM 2 segmentation embeddings to preserve object boundaries and details.
*   **Parallelized CAFB Architecture**: Introduced the *Cross-Attention Fusion Block (CAFB)*. Text, image embeddings and segmentation embeddings are now processed in **parallel** and fused via a trainable convolutional layer, streamlining the information flow compared to sequential approaches.
*   **SAM 2 Perceptual Loss**: Integrated a new perceptual loss function based on the SAM 2 feature space, supplementing the standard MSE diffusion loss to improve semantic consistency in the super-resolved output.
*   **Memory Optimization**: The architecture is optimized for consumer-grade hardware (e.g., NVIDIA RTX 4090), significantly reducing VRAM usage without compromising performance.

## 🛠️ Installation

The environment can be set up using standard Python package management.

1.  **Clone the repository**
    ```bash
    git clone https://github.com/andreaunitn/Super-Resolution.git
    cd Super-Resolution
    ```
    
2.  **Create an environment**
    ```bash
      conda create -n samsr python=3.8
      conda activate samsr
    ```
    
3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

    *Note: If you are using xformers for memory efficiency, ensure it is compatible with your PyTorch/CUDA version.*

## 🚀 Inference
#### Download the pretrained models
- Download the pretrained SD-2-base model from [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-2-base) or [Google Drive](https://drive.google.com/drive/folders/1MaWk4nfuSgBVD5ZqzKKYb5zB5Zo46Ak_?usp=sharing).
- Download the SeeSR, DAPE, RAM, Tiny VAE and SAM 2 models from [GoogleDrive](https://drive.google.com/drive/folders/1EBCcwxXOOcLk9sNMFTjjulWGagQOJS85?usp=sharing).
- Download the test datasets (DIV2K-Val, RealLR200, RealSR and DRealSR) from [Google Drive](https://drive.google.com/drive/folders/1pTVdJnvEIMK2Myj3IFylRooSFdqOUHac?usp=sharing).

You can put the models into `preset/models`, the test datasets into c and then run:
```bash
./scripts/test.sh
```
## 🌈 Train 
#### Step 1: Prepare training data
Pre-prepare training data pairs for the training process, which would take up some memory space but save training time. I trained SamSR with 15% of [LSDIR](https://huggingface.co/ofsoundof/LSDIR) randomly sampled using the `scripts/make_train_subset.sh` script + the first 5K images of [FFHQ](https://huggingface.co/datasets/marcosv/ffhq-dataset). Put the images of LSDIR into `preset/datasets/train_datasets/LSDIR/full_dataset`, execute the script and the add the images from FFHQ into `preset/datasets/train_datasets/LSDIR/finetune_subset`.

For making paired data when training SamSR, you can run:
```
python -W ignore utils_data/make_paired_data.py \
--gt_path preset/datasets/train_datasets/LSDIR/finetune_subset \
--save_dir preset/datasets/train_datasets/LSDIR \
--epoch 1
```

- `--gt_path` the path of gt images. If you have multi gt dirs, you can set it as `PATH1 PATH2 PATH3 ...`
- `--save_dir` the path of paired images 
- `--epoch` the number of epoch you want to make

The difference between `make_paired_data_DAPE.py` and `make_paired_data.py` lies in that `make_paired_data_DAPE.py` resizes the entire image to a resolution of 512, while `make_paired_data.py` randomly crops a sub-image with a resolution of 512.

Include the SAM 2 repository into your own. Then, ror making the data, you can run:
```bash
./scripts/make_seg.sh
```
Once the degraded data pairs and SAM 2 data are created, generate tag data by running `utils_data/make_tags.py`.

The data folder should be like this:
```
your_training_datasets/
    └── gt
        └── 0000001.png # GT images, (512, 512, 3)
        └── ...
    └── lr
        └── 0000001.png # LR images, (512, 512, 3)
        └── ...
    └── tag
        └── 0000001.txt # tag prompts
        └── ...
```

#### Step 2: Training SamSR
```bash
./scripts/train.sh
```

## 📜 Credits & Acknowledgments
This project is built upon the excellent research of **SeeSR** and **SAM 2**.

* **Original Codebase**: [SeeSR (CVPR 2024)](https://github.com/cswry/SeeSR).
* **Segment Anything Model 2**: [Meta AI SAM 2](https://github.com/facebookresearch/sam2).

If you find this code useful, please consider citing the original works:
The following is BibTeX reference:

```
@inproceedings{wu2024seesr,
  title={Seesr: Towards semantics-aware real-world image super-resolution},
  author={Wu, Rongyuan and Yang, Tao and Sun, Lingchen and Zhang, Zhengqiang and Li, Shuai and Zhang, Lei},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={25456--25467},
  year={2024}
}
```
## 🎫 License
This project and related weights are released under the [Apache 2.0 license](LICENSE).
