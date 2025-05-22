# DiffusionReward: Enhancing Blind Face Restoration through Reward Feedback Learning

<p align="center">
    <img src="others/logo.png" width="400">
</p>

## 📖 Overview

This project introduces the DiffusionReward method, which enhances blind face restoration through reward feedback learning. Using DiffBIR+ReFL as an example, we provide complete validation and training code, along with anonymous pre-trained weight files.

## 🚀 Quick Start

### Environment Setup

1. **Create conda environment**
   ```bash
   conda create -n DiffusionReward python=3.10
   conda activate DiffusionReward
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Dataset Preparation

**Test Dataset Download**
- CelebA-Test, LFW-Test, and Webphoto-Test datasets are from the VQFR public repository
- Download link: [VQFR Repository](https://github.com/TencentARC/VQFR)

### Model Weights Download

**Download Pre-trained Weights**
- Download DiffusionReward weight files: [Google Drive Link](https://drive.google.com/drive/folders/1d0ASMR6aH3rtYx9Quyh_eqMbwLKIfgLP?usp=sharing)
- Place the downloaded weight files in the `./weights` directory
- Weight files include:
  - Face Reward Model weights
  - DiffBIR+ReFL weights

**Note**: Please modify the weight paths in the configuration files according to your actual file locations.

## 🔍 Model Validation

Run the following command to perform model validation:

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --task face \
    --diffbir_refl_path ./weights/diffbir_refl.pt \
    --version v1_refl \
    --captioner none \
    --pos_prompt '' \
    --neg_prompt "low quality,blurry,low-resolution,noisy,unsharp,weird textures" \
    --cfg_scale 1.0 \
    --sampler ddim \
    --input ./dataset/celebA/celeba_512_validation_lq/ \
    --output ./output/celeba_diffbir/ \
    --device cuda \
    --precision fp32
```

### Parameter Description

- `--task`: Task type, set to `face`
- `--diffbir_refl_path`: Path to DiffBIR+ReFL model weights
- `--version`: Model version
- `--input`: Input image path
- `--output`: Output result save path
- `--sampler`: Sampler type

## 🏋️ Model Training

### Training Dataset Preparation

1. **FFHQ Dataset Preparation**
   - First download the FFHQ dataset: [FFHQ Dataset](https://github.com/NVlabs/ffhq-dataset)
   - Download our prepared text description files: [Text Description Files](https://drive.google.com/file/d/1rqdeWr_BB50Q-4_1H6AfR8WEf9G31M7_/view?usp=sharing)
   - Extract the text description files to the FFHQ image folder
   - Download the [images_path_list.txt](https://drive.google.com/drive/folders/1d0ASMR6aH3rtYx9Quyh_eqMbwLKIfgLP?usp=sharing) file, which contains relative path information for images

2. **Dataset File Structure**
   ```
   dataset/
   └── FFHQ/
       ├── FFHQ_&_caption/
       │   ├── 00000.png
       │   ├── 00000.txt
       │   ├── ...
       │   ├── 69999.png
       │   └── 69999.txt
       └── image_path_list.txt
   ```

### Start Training

Use the following command to start training:

```bash
accelerate launch train.py --config configs/train/train_diffbir_refl.yaml
```


## 🤝 Contributing

We welcome Issues and Pull Requests to improve this project.

