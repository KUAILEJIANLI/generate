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

        # 3. 将 CenterCrop 改为 RandomCrop 
        # 模拟“切片训练”，大幅增加模型在同一张图上见到的样本多样性
        self.base_transform = T.Compose([
            T.Resize(size + 64), # 略微放大，为随机裁剪留出空间
            T.RandomCrop(size),
        ])

    def __len__(self):
        return len(self.image_paths)

    def degrade(self, img_tensor):
        """
        极端化退化逻辑 (对标论文 4.1.1 节: 模拟信息截断场景)
        目的: 彻底破坏局部像素分布，强迫模型依赖 L-SGB 的结构锚定。
        """
        deg_img = img_tensor.clone()
        
        # A. 极端高斯模糊 (模拟红外镜头严重失焦或运动浓雾)
        if random.random() < 0.7: 
            # 将模糊上限从 3.0 提升至 5.0，造成严重的高频纹理丢失
            sigma = random.uniform(0.5, 5.0) 
            k = int(2 * round(3 * sigma) + 1)
            deg_img = TF.gaussian_blur(deg_img, [k, k], [sigma, sigma])
            
        # B. 极端高斯噪声 (模拟传感器在极端暗光下的热噪声)
        if random.random() < 0.6:
            # 提高噪声强度上限 (0.15 -> 0.4)，彻底淹没像素级梯度
            noise_level = random.uniform(0.05, 0.4) 
            noise = torch.randn_like(deg_img) * noise_level
            deg_img = deg_img + noise
            
        # C. 极端暗光/亮度抖动 (模拟极低照度下的致盲场景)
        if random.random() < 0.5:
            # 范围降至 0.05 (近乎全黑)，测试模型对微弱光子的捕获能力
            brightness_factor = random.uniform(0.05, 0.8) 
            deg_img = deg_img * brightness_factor
            
        # D. 极端随机遮挡 (对标 Mask-DiFuser 的掩码策略)
        # 逻辑: 生成大型黑色遮挡块，强迫模型执行“语义补全”而非“像素平滑”
        if random.random() < 0.4: # 概率从 0.3 提升至 0.4
            h, w = deg_img.shape[-2:]
            # 增加遮挡尺寸上限，模拟大面积信息丢失
            mask_size = random.randint(h // 8, h // 3) 
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