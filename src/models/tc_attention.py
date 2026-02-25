import torch
import torch.nn as nn
import math

class TCRefinementAttention(nn.Module):
    """
    Time-aware Condition Refinement (TC-Refinement) Module
    论文 4.3: 基于时间步 t 动态调节结构与纹理的注入
    """
    def __init__(self, query_dim, context_dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        # 标准的 Q, K, V 映射
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        # === 创新点: 时间感知控制器 ===
        # 输入时间嵌入 time_emb，输出一个缩放因子 alpha
        # 假设 time_emb 维度是 1280 (SD标准)
        self.time_mixer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(1280, inner_dim), 
            nn.Sigmoid() # 输出 (0, 1) 之间的权重
        )

        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context, time_emb):
        """
        x: 来自 U-Net 的隐层特征 (Query) [B, H*W, C]
        context: 来自 L-SGB 的结构特征 (Key/Value) [B, H*W, C_ctx]
        time_emb: 扩散时间步嵌入 [B, 1280]
        """
        h = self.heads
        
        # 1. 计算 Q, K, V
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        # 重塑维度以进行多头注意力
        # [B, Seq, Heads*Dim] -> [B, Heads, Seq, Dim]
        b, seq, _ = q.shape
        q = q.view(b, seq, h, -1).permute(0, 2, 1, 3)
        k = k.view(b, seq, h, -1).permute(0, 2, 1, 3)
        v = v.view(b, seq, h, -1).permute(0, 2, 1, 3)

        # 2. 计算 Attention Score
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # === 创新点: 注入时间感知偏置 (Time-aware Bias) ===
        # 逻辑: 
        # t 很大 (噪声大) -> alpha 倾向于关注粗粒度结构
        # t 很小 (噪声小) -> alpha 倾向于关注细粒度纹理
        
        # 计算时间权重 alpha: [B, Heads*Dim] -> [B, Heads, 1, Dim]
        alpha = self.time_mixer(time_emb) 
        alpha = alpha.view(b, h, 1, -1)
        
        # 这里我们将 alpha 作用在 Value 上 (这是一种特征重加权策略)
        # 意味着: 在不同的时间步，我们从 L-SGB 提取的信息侧重点不同
        v = v * alpha 

        # 3. Softmax 归一化
        attn = dots.softmax(dim=-1)

        # 4. 加权求和
        out = torch.matmul(attn, v)
        
        # 还原维度
        out = out.permute(0, 2, 1, 3).contiguous().view(b, seq, -1)
        return self.to_out(out)