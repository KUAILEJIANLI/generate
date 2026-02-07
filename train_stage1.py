import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 降低碎片化风险
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.sald_model import SALDModel
from src.data.sald_dataset import SyntheticRestorationDataset

CONFIG = {
    "device": "cuda",
    "data_path": "/home/tf/dataset/mini_imageNet", 
    "save_path": "checkpoints/stage1_sald",
    
    # === [绝境配置] ===
    "img_size": 256,       # <--- 核心修改: 从 512 降到 256 (显存占用 -70%)
    "batch_size": 4,       # 4张卡，每张卡跑1个 (Total 4)
    "grad_accum_steps": 8, # 梯度累积，等效 Batch = 32
    "epochs": 10,
    "lr": 1e-5
}

def train():
    os.makedirs(CONFIG['save_path'], exist_ok=True)
    
    # 1. 检测 GPU
    gpu_count = torch.cuda.device_count()
    print(f"⚡ Detected {gpu_count} GPUs. Average VRAM per card: ~10GB (Estimated)")
    
    # 2. 数据 (注意这里的 size)
    print(f"Initializing Dataset (Size={CONFIG['img_size']})...")
    dataset = SyntheticRestorationDataset(CONFIG['data_path'], size=CONFIG['img_size'])
    dataloader = DataLoader(
        dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=4, # 降低 worker 数量以减少 CPU 内存开销
        pin_memory=True,
        drop_last=True
    )
    
    # 3. 模型
    print("Initializing SALD Model...")
    model = SALDModel(device="cuda:0").to("cuda:0")
    
    # =======================================================
    # 🛡️ 显存保卫战 (MAXIMUM SAVING)
    # =======================================================
    
    # [1] 梯度检查点 (必开)
    model.unet.enable_gradient_checkpointing()
    
    # [2] VAE 优化 (必开)
    model.vae.enable_slicing()
    model.vae.enable_tiling()
    
    # [3] Attention 切片 (必开 - 替代 xformers)
    # 如果没有 xformers，这个函数能救命。它把计算拆得非常碎。
    if hasattr(model.unet, "set_attention_slice"):
        model.unet.set_attention_slice("auto")
        print("✅ Attention Slicing enabled (auto)!")
    
    # 尝试开启 xformers (如果有的话更好)
    try:
        model.unet.enable_xformers_memory_efficient_attention()
        print("✅ xFormers also enabled!")
    except:
        pass
        
    # =======================================================
    
    # 4. 优化器
    trainable_params = list(model.l_sgb.parameters()) + \
                       list(model.adapter.parameters()) + \
                       list(model.unet.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=CONFIG['lr'])
    
    # 5. 多卡并行
    if gpu_count > 1:
        print(f"🚀 Activating DataParallel on IDs: {list(range(gpu_count))}")
        model = nn.DataParallel(model)
    
    scaler = GradScaler()

    # 6. 训练
    print("Start Training...")
    model.train()
    
    # DataParallel 下访问 module 属性
    if isinstance(model, nn.DataParallel):
        model.module.vae.eval()
    else:
        model.vae.eval()
    
    for epoch in range(CONFIG['epochs']):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        loss_sum = 0
        
        for step, (ir, vis, gt) in enumerate(loop):
            ir = ir.cuda()
            vis = vis.cuda()
            gt = gt.cuda()
            
            # 清理缓存 (稍微牺牲速度，防止碎片化 OOM)
            # torch.cuda.empty_cache() 
            
            with autocast():
                loss = model(ir, vis, gt)
                
                if gpu_count > 1:
                    loss = loss.mean()
                
                loss = loss / CONFIG['grad_accum_steps']
            
            scaler.scale(loss).backward()
            
            if (step + 1) % CONFIG['grad_accum_steps'] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            current_loss = loss.item() * CONFIG['grad_accum_steps']
            loss_sum += current_loss
            loop.set_postfix(loss=current_loss)
            
        print(f"Epoch {epoch+1} Avg Loss: {loss_sum/len(dataloader):.4f}")
        
        if gpu_count > 1:
            save_dict = model.module.state_dict()
        else:
            save_dict = model.state_dict()
        torch.save(save_dict, os.path.join(CONFIG['save_path'], "sald_stage1_latest.pth"))

if __name__ == "__main__":
    train()