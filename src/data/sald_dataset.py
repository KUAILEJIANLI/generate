import os
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class SyntheticRestorationDataset(Dataset):
    """
    SALD Stage 1 专用数据集:
    任务: 自监督预训练 (Self-Supervised Pre-training)
    逻辑: 输入单张清晰图 -> 在线合成 (伪IR, 伪Vis) + 原始GT
    """
    def __init__(self, root_dir, size=512):
        """
        Args:
            root_dir (str): 图片根目录 (如 ImageNet/train 或 mini_imageNet)
            size (int): 训练分辨率 (Stable Diffusion 默认为 512)
        """
        self.root_dir = root_dir
        self.size = size
        self.image_paths = []
        
        # 1. 扫描所有图片文件
        valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        if os.path.exists(root_dir):
            print(f"Scanning images in {root_dir}...")
            for root, dirs, files in os.walk(root_dir):
                for file in files:
                    if os.path.splitext(file)[1].lower() in valid_ext:
                        self.image_paths.append(os.path.join(root, file))
        
        if not self.image_paths:
            print(f"⚠️ Warning: No images found in {root_dir}")
        else:
            print(f"✅ Found {len(self.image_paths)} training images.")

        # 基础预处理: 调整大小并裁剪
        self.base_transform = T.Compose([
            T.Resize(size, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(size),
        ])

    def __len__(self):
        return len(self.image_paths)

    def degrade(self, img_tensor):
        """
        核心函数: 制造'烂图'来模拟红外/可见光的退化
        Input: Tensor [1, H, W], Range [0, 1]
        """
        # clone 一份，以免修改原图
        deg_img = img_tensor.clone()
        
        # A. 随机高斯模糊 (模拟红外模糊或失焦)
        if random.random() < 0.7: # 70% 概率变模糊
            sigma = random.uniform(0.1, 3.0)
            # kernel size 必须是奇数
            k = int(2 * round(3 * sigma) + 1)
            # 限制 k 最小为 1
            k = max(1, k) 
            deg_img = TF.gaussian_blur(deg_img, [k, k], [sigma, sigma])
            
        # B. 随机高斯噪声 (模拟传感器热噪声)
        if random.random() < 0.6:
            noise_level = random.uniform(0.01, 0.15)
            noise = torch.randn_like(deg_img) * noise_level
            deg_img = deg_img + noise
            
        # C. 随机强度抖动 (模拟光照变化或红外热感差异)
        if random.random() < 0.5:
            brightness_factor = random.uniform(0.5, 1.2)
            deg_img = deg_img * brightness_factor
            
        # D. 随机遮挡/丢失 (强迫模型通过上下文补全)
        # 这对于训练生成模型的 Structure Branch 非常有效
        if random.random() < 0.3:
            h, w = deg_img.shape[-2:]
            mask_size = random.randint(h // 10, h // 4)
            x = random.randint(0, w - mask_size)
            y = random.randint(0, h - mask_size)
            deg_img[:, y:y+mask_size, x:x+mask_size] = 0.0

        # 截断回 [0, 1] 范围
        return torch.clamp(deg_img, 0.0, 1.0)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            # 1. 读取并处理 GT (清晰原图)
            # 强制转为灰度 (因为红外融合通常处理单通道，或者你可以选 RGB)
            # 这里为了适配 VAE，我们先读成 RGB，后面再处理
            img = Image.open(path).convert('RGB')
            
            # 基础尺寸调整
            img = self.base_transform(img)
            
            # 转为 Tensor [3, H, W], Range [0, 1]
            gt_tensor = TF.to_tensor(img)
            
            # 2. 生成伪 IR 和 伪 Vis
            # 先转灰度 [1, H, W] 用于 L-SGB 输入
            gray_tensor = TF.rgb_to_grayscale(gt_tensor)
            
            # 分别进行不同的退化，模拟"两张不同的烂图"
            fake_ir = self.degrade(gray_tensor)   # [1, H, W]
            fake_vis = self.degrade(gray_tensor)  # [1, H, W]
            
            # 3. 数据归一化 (关键步骤!)
            
            # A. 对于条件输入 (L-SGB): Range [0, 1] 是合适的
            # 已经是 [0, 1] 了，无需变动
            
            # B. 对于 GT (VAE 输入): Range 必须是 [-1, 1]
            # Stable Diffusion VAE 严格要求输入在 -1 到 1 之间
            gt_norm = (gt_tensor * 2.0) - 1.0
            
            return fake_ir, fake_vis, gt_norm
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # 出错时返回一个全黑/随机数据，防止训练中断
            dummy_cond = torch.zeros(1, self.size, self.size)
            dummy_gt = torch.zeros(3, self.size, self.size)
            return dummy_cond, dummy_cond, dummy_gt