# DiffusionReward: 通过奖励反馈学习增强盲脸修复

<p align="center">
    <img src="others/logo.png" width="400">
</p>

<p align="center">
    <a href="https://arxiv.org/abs/2505.17910">📄 在 arXiv 上阅读我们的论文</a>
</p>

## 📖 概述

本项目介绍了 DiffusionReward 方法，该方法通过奖励反馈学习来增强盲脸修复。我们以 DiffBIR+ReFL 为例，提供了推理和训练代码。

## 🖼️ 模型架构与结果

### 模型架构
面部奖励模型 (Face Reward Model)
<p align="center">
    <img src="others/frm.jpg" width="600" alt="DiffusionReward Architecture">
</p>

DiffusionReward
<p align="center">
    <img src="others/new_frame.jpg" width="600" alt="DiffusionReward Architecture">
</p>

### 视觉结果
前后对比
<p align="center">
    <img src="others/guidence_show_compressed.jpg" width="800" alt="DiffusionReward Results Comparison">
</p>

## 🚀 快速上手

### 环境配置

1. **创建 conda 环境**
   ```bash
   conda create -n DiffusionReward python=3.10
   conda activate DiffusionReward
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

## 🔍 模型推理

### 数据集准备

**测试数据集下载**
- CelebA-Test, LFW-Test 和 Webphoto-Test 数据集来自 VQFR 公开仓库
- 下载链接: [VQFR Repository](https://github.com/TencentARC/VQFR)

### 模型权重下载
在此 [下载](https://drive.google.com/file/d/1jLZP_NGxKcfhjN8rJXcbtBGXTin4vo3x/view?usp=sharing) Diffbir+ReFL 的 ControlNet 权重。  
将权重文件放置在 `./weights` 文件夹中。

运行以下命令进行模型验证：

### 推理命令
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

### 参数说明

- `--task`: 任务类型，设置为 `face`
- `--diffbir_refl_path`: DiffBIR+ReFL 模型权重的路径
- `--version`: 模型版本
- `--input`: 输入图像路径
- `--output`: 输出结果保存路径
- `--sampler`: 采样器类型

## 🏋️ 模型训练

### 训练数据集准备

1. **FFHQ 数据集准备**
   - 首先下载 FFHQ 数据集: [FFHQ Dataset](https://github.com/NVlabs/ffhq-dataset)
   - 下载我们准备好的文本描述文件: [Text Description Files](https://drive.google.com/file/d/1rqdeWr_BB50Q-4_1H6AfR8WEf9G31M7_/view?usp=sharing)
   - 将文本描述文件解压到 FFHQ 图像文件夹中
   - 下载 [images_paths.txt](https://drive.google.com/file/d/1lip4_0VdIH9L1cZoHw6IJ3_WrDeAFv3k/view?usp=sharing) 文件，其中包含图像的相对路径信息

2. **数据集文件结构**
   ```
   dataset/
   └── FFHQ/
       ├── FFHQ_&_caption/
       │   ├── 00000.png
       │   ├── 00000.txt
       │   ├── ...
       │   ├── 69999.png
       │   └── 69999.txt
       └── image_paths.txt
   ```
### 模型权重下载

要运行训练阶段，请下载以下预训练模型权重并将其放入 `./weights` 文件夹中：

- **Stable Diffusion**: 在此 [下载](https://drive.google.com/file/d/1FFLDPLiHcKA4AQjW6AuVz8RUkkUcLmXs/view?usp=drive_link) SD-2.1 模型的预训练权重。  

- **DiffBIR**: 在此 [下载](https://drive.google.com/file/d/1-8esNaZIVyzM4ttpJg795E4ucpRrvEaC/view?usp=drive_link) DiffBIR-v1 face 模型的预训练权重。  

- **SwinIR**: 在此 [下载](https://drive.google.com/file/d/11T6OsjATJ5gEbkBGOQHRVR38ltTtm8kb/view?usp=drive_link) SwinIR 模型的预训练权重。  

- **FaceReward**: 在此 [下载](https://drive.google.com/file/d/1ugATjemF-70b4N1dQrQJ8J1cdmiVeqOR/view?usp=drive_link) FaceReward 模型的预训练权重。  


### 开始训练

使用以下命令开始训练：

```bash
accelerate launch train.py --config configs/train/train_diffbir_refl.yaml
```
**注意**: 请根据您的实际文件位置修改 `.yaml` 文件中的权重路径。

## 🙏 致谢
本工作构建在 [DiffBIR](https://github.com/XPixelGroup/DiffBIR) 仓库提供的优秀基础之上。我们衷心感谢作者在盲图像修复领域所做的杰出贡献。

## 🤝 引用
如果您使用了我们的代码或认为我们的论文对您有所帮助，请引用我们的论文：
```bash
@misc{wu2025diffusionrewardenhancingblindface,
      title={DiffusionReward: Enhancing Blind Face Restoration through Reward Feedback Learning}, 
      author={Bin Wu and Wei Wang and Yahui Liu and Zixiang Li and Yao Zhao},
      year={2025},
      eprint={2505.17910},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.17910}, 
}
```
