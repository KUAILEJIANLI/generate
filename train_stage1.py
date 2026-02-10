import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32" # 进一步改小碎片阈值

import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed

# 引入 8-bit
try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.models.sald_model import SALDModel
from src.data.sald_dataset import SyntheticRestorationDataset

CONFIG = {
    "data_path": "/home/tf/dataset/mini_imageNet", 
    "save_path": "checkpoints/stage1_sald",
    "batch_size": 1,       
    "img_size": 256,       
    "grad_accum_steps": 4, 
    "epochs": 10,
    "lr": 1e-4,
    "seed": 42
}

def train():
    # === [关键修改] 禁用 cuDNN Benchmark 以节省显存 ===
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # ===============================================

    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=CONFIG['grad_accum_steps']
    )
    set_seed(CONFIG['seed'])

    if accelerator.is_main_process:
        os.makedirs(CONFIG['save_path'], exist_ok=True)
        print(f"🚀 Launching Distributed Training on {accelerator.num_processes} GPUs!")

    # 数据
    dataset = SyntheticRestorationDataset(CONFIG['data_path'], size=CONFIG['img_size'])
    dataloader = DataLoader(
        dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        drop_last=True
    )
    
    # 模型 (现在默认加载 FP16)
    model = SALDModel() 
    
    # 显存优化
    model.vae.enable_slicing()
    model.vae.enable_tiling()
    
    # 尝试开启 xFormers (如果你装了的话，这是省显存神器)
    try:
        model.unet.enable_xformers_memory_efficient_attention()
    except:
        pass # 没装就算了

    if hasattr(model.unet, "enable_gradient_checkpointing"):
        model.unet.enable_gradient_checkpointing()

    # 优化器
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    if HAS_BNB:
        # 8-bit AdamW
        optimizer = bnb.optim.AdamW8bit(params_to_optimize, lr=CONFIG['lr'])
    else:
        optimizer = torch.optim.AdamW(params_to_optimize, lr=CONFIG['lr'])
    
    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )

    if accelerator.is_main_process:
        print("Start Training...")
        
    model.train()
    if hasattr(model, "module"):
        model.module.vae.eval()
    else:
        model.vae.eval()

    for epoch in range(CONFIG['epochs']):
        if accelerator.is_main_process:
            loop = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        else:
            loop = dataloader 
            
        for step, (ir, vis, gt) in enumerate(loop):
            with accelerator.accumulate(model): 
                loss = model(ir, vis, gt)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            
            if accelerator.is_main_process:
                loop.set_postfix(loss=loss.item())

        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), os.path.join(CONFIG['save_path'], "sald_stage1_latest.pth"))
            print(f"✅ Saved checkpoint for Epoch {epoch+1}")

if __name__ == "__main__":
    train()