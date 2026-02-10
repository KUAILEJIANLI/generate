import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig, get_peft_model
from src.models.l_sgb import LSGB
import peft.tuners.lora as lora_module

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
        
        print("Injecting LoRA adapters...")
        lora_config = LoraConfig(
            r=8, 
            lora_alpha=32,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=0.05,
            bias="none"
        )
        self.unet = get_peft_model(self.unet, lora_config)
        
        # 调度器不涉及大显存，保持默认
        self.scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        
        # 自定义组件 L-SGB 和 Adapter 参数量小，建议用 FP32 保持精度
        # 混合精度训练时，autocast 会自动处理它们的计算精度
        self.cross_attention_dim = 768
        self.lsgb_dim = 256
        self.l_sgb = LSGB(in_channels=1, features=[32, 64, 128, self.lsgb_dim])
        
        self.adapter = nn.Sequential(
            nn.Linear(self.lsgb_dim, self.cross_attention_dim),
            nn.SiLU(),
            nn.Linear(self.cross_attention_dim, self.cross_attention_dim)
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
        # 1. Latents
        latents = self.encode_latents(gt)
        
        # 2. Noise & Timesteps
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, 
            (bsz,), device=latents.device
        ).long()
        
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        # 3. L-SGB (FP32 -> FP16 自动转换)
        sgb_feats = self.l_sgb(ir, vis) 
        raw_cond = sgb_feats[-1]
        b, c, h, w = raw_cond.shape
        raw_cond = raw_cond.view(b, c, -1).permute(0, 2, 1) 
        
        # 4. Adapter
        encoder_hidden_states = self.adapter(raw_cond)
        
        # Time-Aware Injection
        t_norm = timesteps.float() / self.scheduler.config.num_train_timesteps
        t_norm = t_norm.view(bsz, 1, 1).to(encoder_hidden_states.dtype)
        encoder_hidden_states = encoder_hidden_states * (0.5 + t_norm)
        
        # 5. U-Net Predict
        # U-Net 是 FP16 的，确保输入也是 FP16 (autocast 会处理，但手动 cast 更稳)
        noisy_latents = noisy_latents.to(self.unet.dtype)
        encoder_hidden_states = encoder_hidden_states.to(self.unet.dtype)
        
        noise_pred = self.unet(
            noisy_latents, 
            timesteps, 
            encoder_hidden_states=encoder_hidden_states
        ).sample
        
        # Loss 计算 (转回 float32 算 Loss 防止溢出)
        loss = F.mse_loss(noise_pred.float(), noise.float())
        return loss