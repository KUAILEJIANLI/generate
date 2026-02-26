import os
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from torchvision.utils import save_image
from datetime import datetime

from src.models.sald_model import SALDModel
from src.data.msrs_dataset import MSRSFusionDataset 
from src.losses.fusion_loss import UnsupervisedFusionLoss

def get_args():
    parser = argparse.ArgumentParser(description="SALD Stage 2 Training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume_from_stage1", type=str, default=None)
    return parser.parse_args()

def train():
    args = get_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    if args.resume_from_stage1:
        cfg['paths']['stage1_weights'] = args.resume_from_stage1

    accelerator = Accelerator(mixed_precision=cfg['train']['mixed_precision'], gradient_accumulation_steps=cfg['train']['grad_accum_steps'])
    
    if accelerator.is_main_process:
        print("\n" + "="*30 + " SALD STAGE 2 START " + "="*30)
        print(yaml.dump(cfg, default_flow_style=False))
        print("="*79 + "\n")

    set_seed(cfg['train']['seed'])
    
    # 路径与数据加载
    save_dir = os.path.join(cfg['paths']['save_path'], datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    preview_dir = os.path.join(save_dir, "previews")
    if accelerator.is_main_process: os.makedirs(preview_dir, exist_ok=True)

    dataset = MSRSFusionDataset(cfg['paths']['msrs_root'], size=tuple(cfg['image']['size']))
    dataloader = DataLoader(dataset, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=cfg['train']['num_workers'])

    # 初始化模型与加载 Stage 1
    model = SALDModel()
    state_dict = torch.load(cfg['paths']['stage1_weights'], map_location="cpu")
    model.load_state_dict(state_dict)
    
    fusion_loss_fn = UnsupervisedFusionLoss(w_int=cfg['loss']['w_int'], w_grad=cfg['loss']['w_grad'])
    # 仅优化 L-SGB, Adapter, TC-Refinement
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=float(cfg['train']['lr']))

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    model.train()
    for epoch in range(cfg['train']['epochs']):
        run_total, run_int, run_grad = 0.0, 0.0, 0.0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}") if accelerator.is_main_process else dataloader
        
        for step, (ir, vis, fname) in enumerate(loop):
            ir, vis = ir.to(accelerator.device), vis.to(accelerator.device)
            
            with accelerator.accumulate(model):
                unwrapped_model = accelerator.unwrap_model(model)
                
                # 核心改进：引入 30% 红外先验，强制模型打破可见光依赖
                mixed_input = 0.3 * ir + 0.7 * vis
                latents = unwrapped_model.encode_latents(mixed_input.repeat(1, 3, 1, 1))
                
                # 采样与预测
                timesteps = torch.randint(cfg['diffusion']['t_min'], cfg['diffusion']['t_max'], (latents.shape[0],), device=latents.device).long()
                noise = torch.randn_like(latents)
                noisy_latents = unwrapped_model.scheduler.add_noise(latents, noise, timesteps)
                
                # 模块协作：L-SGB -> Adapter -> TC-Refinement
                sgb_feats = unwrapped_model.l_sgb(ir.float(), vis.float()) 
                cond = sgb_feats.view(sgb_feats.shape[0], sgb_feats.shape[1], -1).permute(0, 2, 1)
                refined_cond = unwrapped_model.tc_refinement(unwrapped_model.adapter(cond), unwrapped_model.adapter(cond), 
                                                           unwrapped_model.time_proj(timesteps.float().view(-1, 1) / 1000.0))
                
                noise_pred = unwrapped_model.unet(noisy_latents.to(unwrapped_model.unet.dtype), timesteps, 
                                               encoder_hidden_states=refined_cond.to(unwrapped_model.unet.dtype)).sample
                
                # 解码融合图像
                alpha_t = unwrapped_model.scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1).to(latents.device)
                pred_x0 = torch.clamp((noisy_latents - (1 - alpha_t)**0.5 * noise_pred) / (alpha_t**0.5), -4.0, 4.0)
                fusion_imgs = unwrapped_model.vae.decode((pred_x0 / 0.18215).to(unwrapped_model.vae.dtype)).sample
                # 解码后转回 float 以便进行稳定的损失计算
                fusion_imgs = (fusion_imgs.float() + 1.0) / 2.0
                loss, l_int, l_grad = fusion_loss_fn(fusion_imgs, ir, vis)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                run_total += loss.item(); run_int += l_int.item(); run_grad += l_grad.item()
                
            if accelerator.is_main_process:
                loop.set_postfix(total=f"{run_total/(step+1):.4f}", int=f"{run_int/(step+1):.4f}", grad=f"{run_grad/(step+1):.4f}")
                if step % 100 == 0:
                    save_image(torch.cat([ir.repeat(1,3,1,1), vis.repeat(1,3,1,1), fusion_imgs], dim=3), f"{preview_dir}/e{epoch+1}_s{step}.png", nrow=1)

        if accelerator.is_main_process:
            print(f"📊 Epoch {epoch+1} summary: Total={run_total/len(dataloader):.5f}, Int={run_int/len(dataloader):.5f}, Grad={run_grad/len(dataloader):.5f}")
            torch.save(unwrapped_model.state_dict(), f"{save_dir}/sald_stage2_e{epoch+1}.pth")

if __name__ == "__main__": train()