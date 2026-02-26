import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32" # 进一步改小碎片阈值

import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from torchvision.utils import save_image
import time
from datetime import datetime
# 引入 8-bit
try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

import kagglehub
path = kagglehub.dataset_download("ifigotin/imagenetmini-1000")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.models.sald_model import SALDModel
from src.data.sald_dataset import SyntheticRestorationDataset

CONFIG = {
    "data_path": path, 
    "save_path": "checkpoints/stage1_sald",
    "batch_size": 32,       
    "img_size": 256,       
    "grad_accum_steps": 1, 
    "epochs": 100,
    "lr": 1e-4,
    "seed": 42
}

# 1. 生成带时间的子路径（推荐格式：年-月-日_时-分-秒，避免特殊字符）
# 时间格式可自定义，比如 "%Y%m%d_%H%M%S" 是纯数字格式
time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# 2. 拼接新的save_path：原路径 + 时间子路径
original_save_path = CONFIG["save_path"]
new_save_path = os.path.join(original_save_path, time_str)

# 3. 使用CONFIG.update()更新配置（核心操作）
CONFIG.update({"save_path": new_save_path})

@torch.no_grad()
def save_preview(model, dataloader, epoch, step, save_path, accelerator):
    """
    保存训练预览图：展示 IR、Vis、GT 以及模型当前的预测结果 (x0)。
    此函数用于直观监控 Stage 1 训练过程中模型对极端退化图像的修复能力。
    涉及到中文的地方已转为简体中文。
    """
    # 1. 切换到评估模式，防止 Dropout 等层干扰
    model.eval()
    unwrapped_model = accelerator.unwrap_model(model)
    
    # 2. 获取一组预览数据
    try:
        # 获取一个 batch 的数据用于预览
        batch = next(iter(dataloader))
        ir, vis, gt = batch
    except Exception as e:
        if accelerator.is_main_process:
            print(f"⚠️ 预览图保存失败，无法获取数据: {e}")
        model.train()
        return

    # 仅取前 4 张样本（若不足 4 张则取全部），防止预览图过大导致保存缓慢
    num_samples = min(ir.shape[0], 4)
    ir = ir[:num_samples].to(accelerator.device)
    vis = vis[:num_samples].to(accelerator.device)
    gt = gt[:num_samples].to(accelerator.device)

    # 3. 模拟推理过程：预测原图 x0
    # 步骤 A: 将 GT 编码为 Latent 空间特征 (VAE 期望范围为 [-1, 1])
    latents = unwrapped_model.encode_latents(gt) # 形状 [B, 4, H/8, W/8]
    
    # 步骤 B: 采样一个固定的中间时间步 (例如 500)，观察中等强度噪声下的还原效果
    # 相比随机采样，固定时间步更能体现模型随 Epoch 增长而产生的性能提升
    timesteps = torch.tensor([500] * num_samples, device=latents.device).long()
    noise = torch.randn_like(latents)
    noisy_latents = unwrapped_model.scheduler.add_noise(latents, noise, timesteps)

    # 步骤 C: 提取结构特征并进行时变精炼 (复用 SALD 核心逻辑)
    # 根据你最新的修改，l_sgb 现在直接返回最终的特征 Tensor
    sgb_feats = unwrapped_model.l_sgb(ir.float(), vis.float()) 
    raw_cond = sgb_feats 
    
    b, c, h, w = raw_cond.shape
    raw_cond_flat = raw_cond.view(b, c, -1).permute(0, 2, 1) # [B, Seq, 256]

    # 映射并精炼特征 (768 维)
    aligned_cond = unwrapped_model.adapter(raw_cond_flat)
    
    # 生成预览对应的时间步嵌入
    t_val = timesteps.float().view(num_samples, 1) / 1000.0
    t_emb = unwrapped_model.time_proj(t_val)
    
    refined_cond = unwrapped_model.tc_refinement(
        x=aligned_cond, 
        context=aligned_cond,
        time_emb=t_emb
    )

    # 4. 调用 U-Net 预测噪声 (使用混合精度以匹配训练环境)
    with torch.amp.autocast('cuda', enabled=(accelerator.mixed_precision != "no")):
        noise_pred = unwrapped_model.unet(
            noisy_latents.to(unwrapped_model.unet.dtype), 
            timesteps, 
            encoder_hidden_states=refined_cond.to(unwrapped_model.unet.dtype)
        ).sample

    # 5. 根据扩散去噪公式反推原始样本 x0 (Original Sample)
    # 公式: x0 = (x_t - sqrt(1 - alpha_bar_t) * epsilon) / sqrt(alpha_bar_t)
    alpha_prod_t = unwrapped_model.scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1).to(latents.device)
    
    pred_latents = (noisy_latents - (1 - alpha_prod_t) ** 0.5 * noise_pred) / (alpha_prod_t ** 0.5)
    
    # 1. 修复精度不匹配错误
    # 确保 pred_latents 转换为 VAE 的精度 (通常是 float16)
    pred_latents_input = (pred_latents / 0.18215).to(unwrapped_model.vae.dtype)

    # 2. VAE 解码预测出的 Latent 到像素空间
    pred_imgs = unwrapped_model.vae.decode(pred_latents_input).sample # 输出范围约为 [-1, 1]
    
    

    # 7. 拼接对比图并保存
    # 将图像统一转换回 [0, 1] 范围以便展示
    gt_display = (gt + 1.0) / 2.0
    pred_display = (pred_imgs + 1.0) / 2.0
    
    # 将单通道的 IR 和 Vis 条件图扩展为 3 通道，方便横向拼接
    ir_display = ir.repeat(1, 3, 1, 1)
    vis_display = vis.repeat(1, 3, 1, 1)

    # 每行图片排布：[退化红外, 退化可见光, 原始真值GT, 模型当前预测结果]
    # dim=3 是在宽度方向拼接
    comparison = torch.cat([ir_display, vis_display, gt_display, pred_display], dim=3)
    
    # 构造文件名
    save_name = f"preview_e{epoch+1}_s{step:04d}.png"
    save_full_path = os.path.join(save_path, save_name)
    
    # nrow=1 表示每一行显示一个样本及其对应的四项对比
    save_image(comparison.float(), save_full_path, nrow=1, normalize=False)
    
    if accelerator.is_main_process:
        print(f"📸 预览图已成功保存: {save_full_path}")
    
    # 恢复模型至训练模式
    model.train()

def train():
    # === [关键修改] 禁用 cuDNN Benchmark 以节省显存 ===
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # ===============================================
    # 针对 L-SGB 中未参与 loss 计算的多尺度特征层 
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=CONFIG['grad_accum_steps'],
        kwargs_handlers=[ddp_kwargs] # 注入配置
    )
    set_seed(CONFIG['seed'])

    preview_save_path = os.path.join(CONFIG['save_path'], "previews")

    if accelerator.is_main_process:
        os.makedirs(CONFIG['save_path'], exist_ok=True)
        os.makedirs(preview_save_path, exist_ok=True)
        print(f"🚀 Launching Distributed Training on {accelerator.num_processes} GPUs!")

    # 数据
    dataset = SyntheticRestorationDataset(CONFIG['data_path'], size=CONFIG['img_size'])
    dataloader = DataLoader(
        dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=14, 
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


    # 遍历每个epoch
    for epoch in range(CONFIG['epochs']):
        # -------- 新增：初始化epoch级别的loss统计变量 --------
        epoch_loss_sum = 0.0  # 累加当前epoch的所有loss
        epoch_step_count = 0   # 统计当前epoch的step数
        
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
            
            # -------- 新增：累加loss和step数 --------
            # 只在主进程统计（避免多进程重复累加）
            if accelerator.is_main_process:
                epoch_loss_sum += loss.item()  # 累加当前step的loss值
                epoch_step_count += 1  # step数+1
            
            # --- 原有监控代码 ---
            if step % 500 == 0 and accelerator.is_main_process:
                save_preview(model, dataloader, epoch, step, preview_save_path, accelerator)
            # ------------------
            
            if accelerator.is_main_process:
                loop.set_postfix(loss=loss.item())

        # -------- 新增：epoch结束时计算并打印平均loss --------
        if accelerator.is_main_process:
            # 计算平均loss（避免除以0）
            if epoch_step_count > 0:
                epoch_avg_loss = epoch_loss_sum / epoch_step_count
            else:
                epoch_avg_loss = 0.0
            
            # 格式化打印，保留4位小数更易读
            print(f"\n📊 Epoch {epoch+1} finished | Average Loss: {epoch_avg_loss:.4f}")
            
            # 原有保存checkpoint代码
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), os.path.join(CONFIG['save_path'], "sald_stage1_latest.pth"))
            print(f"✅ Saved checkpoint for Epoch {epoch+1}\n")

if __name__ == "__main__":
    train()