import torch
import torch.nn as nn
import torch.nn.functional as F

class StructureBlock(nn.Module):
    """简单的卷积块: Conv-BN-SiLU"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU()
        )
    def forward(self, x):
        return self.conv(x)

class LSGB(nn.Module):
    """
    Lightweight Structure-Guided Branch (L-SGB)
    论文 4.2.2: 提取低频拓扑骨架，提供结构锚定。
    """
    def __init__(self, in_channels=1, features=[32, 64, 128, 256]):
        super().__init__()
        
        # 1. 双流特征提取 (Two-stream extraction)
        # 即使是单通道输入，我们为了鲁棒性，先提取浅层特征
        self.ir_stem = StructureBlock(in_channels, features[0])
        self.vis_stem = StructureBlock(in_channels, features[0])
        
        # 2. 下采样编码器 (提取多尺度结构)
        self.encoders = nn.ModuleList()
        for i in range(len(features)-1):
            self.encoders.append(
                nn.Sequential(
                    nn.MaxPool2d(2),
                    StructureBlock(features[i], features[i+1])
                )
            )
            
        # 3. 结构特征融合层 (1x1 Conv)
        # 用于将 IR 和 Vis 的特征合并
        self.fusion_convs = nn.ModuleList([
            nn.Conv2d(f * 2, f, 1) for f in features
        ])

    def forward(self, ir, vis):
        """
        输入: ir, vis [B, 1, H, W]
        输出: 多尺度结构特征列表 (用于通过 CrossAttention 注入 U-Net)
        """
        # 第一层特征
        f_ir = self.ir_stem(ir)
        f_vis = self.vis_stem(vis)
        
        # Max-Feature Selection (结构锚定核心)
        # 论文逻辑: 在极端退化下，谁的特征强(数值大)，就信谁
        # 这种操作对噪声鲁棒，因为我们取的是"结构响应"的最大值
        f_cat = torch.cat([f_ir, f_vis], dim=1) # [B, 2*C, H, W]
        f_anchored = self.fusion_convs[0](f_cat) # [B, C, H, W]
        
        structure_feats = [f_anchored]
        
        # 下采样提取多尺度特征
        curr_ir, curr_vis = f_ir, f_vis
        for i, enc in enumerate(self.encoders):
            curr_ir = enc(curr_ir)
            curr_vis = enc(curr_vis)
            
            # 同样执行 Max-Fusion 逻辑的变体（这里简化为 Concat+Conv）
            f_cat = torch.cat([curr_ir, curr_vis], dim=1)
            f_anchored = self.fusion_convs[i+1](f_cat)
            
            structure_feats.append(f_anchored)
            
        return structure_feats[-1]