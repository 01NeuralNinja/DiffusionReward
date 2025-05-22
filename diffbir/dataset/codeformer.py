from typing import Sequence, Dict, Union, List, Mapping, Any, Optional
import math
import time
import io
import random
import os
import numpy as np
import cv2
from PIL import Image
import torch.utils.data as data

from .degradation import (
    random_mixed_kernels,
    random_add_gaussian_noise,
    random_add_jpg_compression,
)
from .utils import load_file_list, center_crop_arr, random_crop_arr
from ..utils.common import instantiate_from_config


class CodeformerDataset(data.Dataset):

    def __init__(
        self,
        file_list: str,
        file_backend_cfg: Mapping[str, Any],
        out_size: int,
        crop_type: str,
        blur_kernel_size: int,
        kernel_list: Sequence[str],
        kernel_prob: Sequence[float],
        blur_sigma: Sequence[float],
        downsample_range: Sequence[float],
        noise_range: Sequence[float],
        jpeg_range: Sequence[int],
    ) -> "CodeformerDataset":
        super(CodeformerDataset, self).__init__()
        self.file_list = file_list
        self.image_files = load_file_list(file_list)
        self.file_backend = instantiate_from_config(file_backend_cfg)
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]
        # degradation configurations
        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        self.downsample_range = downsample_range
        self.noise_range = noise_range
        self.jpeg_range = jpeg_range

    def load_gt_image(
        self, image_path: str, max_retry: int = 5
    ) -> Optional[np.ndarray]:
        image_bytes = None
        while image_bytes is None:
            if max_retry == 0:
                return None
            image_bytes = self.file_backend.get(image_path)
            max_retry -= 1
            if image_bytes is None:
                time.sleep(0.5)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if self.crop_type != "none":
            if image.height == self.out_size and image.width == self.out_size:
                image = np.array(image)
            else:
                if self.crop_type == "center":
                    image = center_crop_arr(image, self.out_size)
                elif self.crop_type == "random":
                    image = random_crop_arr(image, self.out_size, min_crop_frac=0.7)
        else:
            assert image.height == self.out_size and image.width == self.out_size
            image = np.array(image)
        # hwc, rgb, 0,255, uint8
        return image

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # load gt image
        img_gt = None
        while img_gt is None:
            # load meta file
            image_file = self.image_files[index]
            gt_path = image_file["image_path"]
            prompt = image_file["prompt"]
            img_gt = self.load_gt_image(gt_path)
            if img_gt is None:
                print(f"filed to load {gt_path}, try another image")
                index = random.randint(0, len(self) - 1)

            # 生成对应的caption文件路径
            caption_path = gt_path.replace('.png', '.txt')

            # 检查caption文件是否存在
            if not os.path.exists(caption_path):
                raise FileNotFoundError(f"Caption文件不存在: {caption_path}")

            # 读取caption内容
            with open(caption_path, 'r') as f:
                caption = f.read().strip()

        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        img_gt = (img_gt[..., ::-1] / 255.0).astype(np.float32)
        h, w, _ = img_gt.shape
        if np.random.uniform() < 0.5:
            prompt = ""

        # ------------------------ generate lq image ------------------------ #
        # blur
        kernel = random_mixed_kernels(
            self.kernel_list,
            self.kernel_prob,
            self.blur_kernel_size,
            self.blur_sigma,
            self.blur_sigma,
            [-math.pi, math.pi],
            noise_range=None,
        )
        img_lq = cv2.filter2D(img_gt, -1, kernel)
        # downsample
        scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        img_lq = cv2.resize(
            img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR
        )
        # noise
        if self.noise_range is not None:
            img_lq = random_add_gaussian_noise(img_lq, self.noise_range)
        # jpeg compression
        if self.jpeg_range is not None:
            img_lq = random_add_jpg_compression(img_lq, self.jpeg_range)

        # resize to original size
        img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)

        # BGR to RGB, [-1, 1]
        gt = (img_gt[..., ::-1] * 2 - 1).astype(np.float32)
        # BGR to RGB, [0, 1]
        lq = img_lq[..., ::-1].astype(np.float32)

        return gt, lq, prompt, caption

    def __len__(self) -> int:
        return len(self.image_files)


class PairedDataset(data.Dataset):
    def __init__(self):
        super(PairedDataset, self).__init__()
        self.lr_path = './dataset/FFHQ/100_LOW_full/'
        self.gt_path = './dataset/FFHQ/100_GT_full/'
        self.out_size = 512

        # 获取图像文件列表
        self.lr_images = sorted([f for f in os.listdir(self.lr_path) if f.endswith(('png', 'jpg', 'jpeg', 'JPEG', 'PNG'))])
        self.gt_images = sorted([f for f in os.listdir(self.gt_path) if f.endswith(('png', 'jpg', 'jpeg', 'JPEG', 'PNG'))])

        # 检查lr和gt图像数量
        assert len(self.lr_images) == len(self.gt_images), "LR和GT图像数量不匹配！"

        # 检查文件名是否完全相同
        for lr_name, gt_name in zip(self.lr_images, self.gt_images):
            assert lr_name == gt_name, f"图像文件名不匹配: LR={lr_name}, GT={gt_name}"

    def __getitem__(self, index):
        # 读取lr图像
        lr_img_path = os.path.join(self.lr_path, self.lr_images[index])
        lr_img = Image.open(lr_img_path).convert('RGB')
        lr_img = np.array(lr_img)

        # 读取gt图像
        gt_img_path = os.path.join(self.gt_path, self.gt_images[index])
        gt_img = Image.open(gt_img_path).convert('RGB')
        gt_img = np.array(gt_img)

        # 如果指定了输出尺寸，进行裁剪
        if self.out_size:
            if gt_img.shape[0] != self.out_size or gt_img.shape[1] != self.out_size:
                gt_img = center_crop_arr(gt_img, self.out_size)
            if lr_img.shape[0] != self.out_size or lr_img.shape[1] != self.out_size:
                lr_img = center_crop_arr(lr_img, self.out_size)

        # 转换为float32并归一化到[-1, 1]
        gt = (gt_img / 255.0 * 2 - 1).astype(np.float32)
        lq = lr_img / 255.0


        lq = (lq * 2 - 1).astype(np.float32)

        # 执行clamp操作
        lq = np.clip(lq, -1, 1)

        return {
            "output_pixel_values": gt,
            "conditioning_pixel_values": lq,
            "prompt": "",
            "neg_prompt": "blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth",
        }

    def __len__(self):
        return len(self.lr_images)