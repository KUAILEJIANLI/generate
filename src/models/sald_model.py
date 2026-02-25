import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig, get_peft_model
from src.models.l_sgb import LSGB
import peft.tuners.lora as lora_module
from src.models.tc_attention import TCRefinementAttention
# === 全局补丁保持不变 ===
def apply_global_lora_patch():
    TargetLayer = None
    if hasattr(lora_module, "Linear"):
        TargetLayer = lora_module.Linear
    elif hasattr(lora_module, "LoraLayer"):
        TargetLayer = lora_module.LoraLayer
    
    if TargetLayer is not None:
        if not hasattr(TargetLayer, "_original_forward_backup"):
            TargetLayer._original_forward_backup = TargetLayer.forward
            def new_forward(self, x, scale=None):
                return self._original_forward_backup(x)
            TargetLayer.forward = new_forward
apply_global_lora_patch()
# ======================

class SALDModel(nn.Module):
    def __init__(self): # 注意：这里去掉了 device 参数，交给 accelerate 管理
        super().__init__()
        
        # [关键修改] 指定加载精度为 float16
        # 这能直接节省 50% 的模型权重显存 (约 2GB)
        dtype = torch.float16 
        model_id = "runwayml/stable-diffusion-v1-5"

        print(f"Loading VAE (FP16)...")
        self.vae = AutoencoderKL.from_pretrained(
            model_id, 
            subfolder="vae",
            torch_dtype=dtype # <--- 关键
        )
        self.vae.requires_grad_(False)
        
        print(f"Loading U-Net (FP16)...")
        self.unet = UNet2DConditionModel.from_pretrained(
            model_id, 
            subfolder="unet",
            torch_dtype=dtype # <--- 关键
        )
        self.unet.requires_grad_(False)
        
        
        # 关键修正：重新设计特征流维度
        self.cross_attention_dim = 768
        self.lsgb_dim = 256
        self.time_emb_dim = 1280

        self.l_sgb = LSGB(in_channels=1, features=[32, 64, 128, self.lsgb_dim])
        
        # 1. 先用 Adapter 把 256 映射到 768
        self.adapter = nn.Sequential(
            nn.Linear(self.lsgb_dim, self.cross_attention_dim),
            nn.SiLU(),
            nn.Linear(self.cross_attention_dim, self.cross_attention_dim)
        )

        # 2. 再用 TC-Refinement 处理 768 维特征
        self.tc_refinement = TCRefinementAttention(
            query_dim=768, 
            context_dim=768, # 修正为与 adapter 输出一致
            heads=8
        )
        
        # 3. 时间步转换模块 (简单版，用于生成 tc_refinement 所需的 time_emb)
        self.time_proj = nn.Sequential(
            nn.Linear(1, 256),
            nn.SiLU(),
            nn.Linear(256, self.time_emb_dim)
        )
        
    def encode_latents(self, img):
        if img.shape[1] == 1: img = img.repeat(1, 3, 1, 1)
        
        # 确保输入和模型在同一精度
        img = img.to(dtype=self.vae.dtype)
        
        with torch.no_grad():
            latents = self.vae.encode(img).latent_dist.sample()
            latents = latents * 0.18215 
        return latents

    def forward(self, ir, vis, gt):
        latents = self.encode_latents(gt)
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, 1000, (bsz,), device=latents.device).long()
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # A. 提取结构锚定特征 [B, 256, H, W]
        sgb_feats = self.l_sgb(ir, vis) 
        raw_cond = sgb_feats[-1] 
        b, c, h, w = raw_cond.shape
        raw_cond = raw_cond.view(b, c, -1).permute(0, 2, 1) # [B, Seq, 256]

        # B. 维度映射 256 -> 768
        aligned_cond = self.adapter(raw_cond)

        # C. 时间感知调制
        # 将 timesteps 归一化并转为 embedding
        t_val = timesteps.float().view(bsz, 1) / 1000.0
        t_emb = self.time_proj(t_val)
        
        # 精炼特征 [B, Seq, 768]
        # x 使用 aligned_cond 作为基础查询，实现自精炼
        refined_cond = self.tc_refinement(
            x=aligned_cond, 
            context=aligned_cond,
            time_emb=t_emb
        )

        # D. U-Net 预测
        noise_pred = self.unet(
            noisy_latents.to(self.unet.dtype), 
            timesteps, 
            encoder_hidden_states=refined_cond.to(self.unet.dtype)
        ).sample
        
        return F.mse_loss(noise_pred.float(), noise.float())