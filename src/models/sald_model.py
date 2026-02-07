import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler

# 引用本地模块
# 确保 src/models/l_sgb.py 已经存在
from src.models.l_sgb import LSGB

class SALDModel(nn.Module):
    """
    SALD: Structure-Anchored Latent Diffusion Model
    基于结构锚定与时变精炼的生成式图像融合模型
    
    架构组成:
    1. VAE (Frozen): 将图像压缩到 Latent Space (降低计算量)
    2. L-SGB: 提取红外与可见光的结构特征
    3. Adapter: 将结构特征投影对齐到 U-Net 的输入维度
    4. U-Net: 标准 Stable Diffusion U-Net (Trainable)
    """
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
        # ====================================================
        # 1. 加载预训练组件 (VAE & U-Net & Scheduler)
        # ====================================================
        model_id = "runwayml/stable-diffusion-v1-5"
        
        print(f"Loading VAE from {model_id} (Frozen)...")
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        # 彻底冻结 VAE，不参与训练，只做特征压缩
        self.vae.requires_grad_(False) 
        
        print(f"Loading U-Net from {model_id}...")
        self.unet = UNet2DConditionModel.from_pretrained(
            model_id, 
            subfolder="unet",
            # 如果你的显存非常紧张(比如<12G)，可以开启 gradient_checkpointing
            # use_gradient_checkpointing=True 
        )
        
        self.scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        
        # ====================================================
        # 2. 自定义组件 (L-SGB & Adapter)
        # ====================================================
        # SD v1.5 的 Cross-Attention 输入维度通常是 768
        self.cross_attention_dim = self.unet.config.cross_attention_dim 
        
        # 初始化 L-SGB (假设输出 256 通道)
        self.lsgb_dim = 256
        self.l_sgb = LSGB(in_channels=1, features=[32, 64, 128, self.lsgb_dim])
        
        # 维度适配器: [B, H*W, 256] -> [B, H*W, 768]
        # 用于将 L-SGB 的特征“翻译”成 U-Net 能听懂的语言
        self.adapter = nn.Sequential(
            nn.Linear(self.lsgb_dim, self.cross_attention_dim),
            nn.SiLU(),
            nn.Linear(self.cross_attention_dim, self.cross_attention_dim)
        )

    def encode_latents(self, img):
        """
        辅助函数: 将 RGB/灰度图压缩为 VAE Latents
        Input: [B, 3, H, W] or [B, 1, H, W] -> Output: [B, 4, H/8, W/8]
        """
        # 适配通道数
        if img.shape[1] == 1:
            img = img.repeat(1, 3, 1, 1)
            
        # VAE 需要输入在 [-1, 1] 之间，如果 DataLoader 输出是 [0, 1]，需要转换
        # img = 2.0 * img - 1.0 
        # (通常在 Dataset 里做 normalize，这里假设输入已经是合理的分布)
            
        with torch.no_grad():
            # 编码 -> 采样 -> 缩放
            # 0.18215 是 SD 官方的缩放因子，用于保持方差稳定
            latents = self.vae.encode(img).latent_dist.sample()
            latents = latents * 0.18215 
        return latents

    def forward(self, ir, vis, gt):
        """
        训练前向传播
        Args:
            ir (Tensor): 红外图像 [B, 1, H, W]
            vis (Tensor): 可见光图像 [B, 1, H, W]
            gt (Tensor): Ground Truth 目标图像 [B, 1, H, W] (或者是 fake_gt)
        Returns:
            loss (Tensor): MSE Loss
        """
        # 1. 准备 Latent (Ground Truth)
        latents = self.encode_latents(gt)
        
        # 2. 采样噪声与时间步
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        
        # 随机采样 t ~ [0, 1000)
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, 
            (bsz,), device=latents.device
        ).long()
        
        # 3. 加噪过程 (Forward Diffusion)
        # noisy_latents = latents * sqrt(alpha_bar) + noise * sqrt(1-alpha_bar)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        # 4. 提取结构条件 (L-SGB)
        # l_sgb 返回的是多尺度特征列表，我们取最深层特征作为 Condition
        sgb_feats = self.l_sgb(ir, vis) 
        raw_cond = sgb_feats[-1] # [B, 256, H/8, W/8]
        
        # 5. 维度变换与投影
        # [B, C, H, W] -> [B, H*W, C]
        b, c, h, w = raw_cond.shape
        raw_cond = raw_cond.view(b, c, -1).permute(0, 2, 1) 
        
        # 通过 Adapter 映射到 768 维
        encoder_hidden_states = self.adapter(raw_cond) # [B, Seq_Len, 768]
        
        # ==========================================================
        # ✨ 创新点实现: 时变精炼 (Time-Aware Refinement)
        # ==========================================================
        # 论文逻辑: 在扩散初期(t大)，结构引导应该更强；后期(t小)，更关注纹理细节。
        # 实现方法: 动态调整 Condition 的强度，而不是去改 U-Net 源码。
        
        # 归一化时间步: t_norm range [0, 1]
        t_norm = timesteps.float() / self.scheduler.config.num_train_timesteps
        t_norm = t_norm.view(bsz, 1, 1) # reshape for broadcasting
        
        # 线性调制策略: weight = 0.5 + t_norm
        # 当 t=1000 (噪声最大) -> weight = 1.5 (增强结构引导)
        # 当 t=0    (噪声最小) -> weight = 0.5 (减弱结构引导，让模型自由发挥纹理)
        time_aware_weight = 0.5 + t_norm
        
        # 将权重注入到条件特征中
        encoder_hidden_states = encoder_hidden_states * time_aware_weight
        
        # ==========================================================
        
        # 6. U-Net 预测噪声
        # 输入: 噪声图, 时间步, 时变结构条件
        noise_pred = self.unet(
            noisy_latents, 
            timesteps, 
            encoder_hidden_states=encoder_hidden_states
        ).sample
        
        # 7. 计算损失
        # 预测噪声 vs 真实噪声
        loss = F.mse_loss(noise_pred, noise)
        
        return loss

    @torch.no_grad()
    def inference(self, ir, vis, steps=50):
        """
        推理/生成函数 (用于验证效果)
        """
        # 1. 提取条件
        sgb_feats = self.l_sgb(ir, vis)
        raw_cond = sgb_feats[-1]
        b, c, h, w = raw_cond.shape
        raw_cond = raw_cond.view(b, c, -1).permute(0, 2, 1)
        encoder_hidden_states = self.adapter(raw_cond)
        
        # 推理时，我们通常使用固定的强引导，或者也可以加入时变逻辑
        # 这里简单起见，使用平均强度 1.0
        
        # 2. 初始化随机噪声 Latent
        latents = torch.randn(
            (b, self.unet.config.in_channels, h, w),
            device=self.device
        )
        
        # 3. 设置调度器步数
        self.scheduler.set_timesteps(steps)
        
        # 4. 逐步去噪循环
        for t in self.scheduler.timesteps:
            # 扩展 Latent 以适配 classifier-free guidance (如果需要的话)
            # 这里是单条件引导，不需要扩展
            
            # 注入时变权重 (可选，保持训练一致性)
            # t_norm = t / 1000.0
            # weight = 0.5 + t_norm
            # current_cond = encoder_hidden_states * weight
            
            # 预测噪声
            noise_pred = self.unet(
                latents, 
                t, 
                encoder_hidden_states=encoder_hidden_states
            ).sample
            
            # 更新 Latent (x_t -> x_{t-1})
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
        # 5. 解码 Latent -> 图像
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        
        # 归一化到 [0, 1] 用于显示
        image = (image / 2 + 0.5).clamp(0, 1)
        return image