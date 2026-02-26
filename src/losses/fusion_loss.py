import torch
import torch.nn as nn
import torch.nn.functional as F

class SobelOperator(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        kernel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        self.register_buffer('weight_x', torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0))
        self.register_buffer('weight_y', torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        # 修复边缘黑色发散：先进行边缘复制填充
        x_pad = F.pad(x, (1, 1, 1, 1), mode='replicate')
        grad_x = F.conv2d(x_pad, self.weight_x.to(x), padding=0)
        grad_y = F.conv2d(x_pad, self.weight_y.to(x), padding=0)
        return torch.abs(grad_x) + torch.abs(grad_y)

class UnsupervisedFusionLoss(nn.Module):
    def __init__(self, w_int=10.0, w_grad=5.0):
        super().__init__()
        self.sobel = SobelOperator()
        self.w_int = w_int
        self.w_grad = w_grad

    def forward(self, fusion_img, ir, vis):
        fusion_img, ir, vis = fusion_img.float(), ir.float(), vis.float()
        fusion_gray = 0.299 * fusion_img[:, 0:1] + 0.587 * fusion_img[:, 1:2] + 0.114 * fusion_img[:, 2:3]
        
        # 1. 显著性亮度损失：红外越亮的地方，惩罚权重越大
        target_int = torch.max(ir, vis)
        saliency_weight = ir + 0.1 
        l_int = torch.mean(saliency_weight * torch.abs(fusion_gray - target_int))
        
        # 2. 梯度损失
        grad_f = self.sobel(fusion_gray)
        grad_ir = self.sobel(ir)
        grad_vis = self.sobel(vis)
        target_grad = torch.max(grad_ir, grad_vis)
        l_grad = F.l1_loss(grad_f, target_grad)
        
        total_loss = self.w_int * l_int + self.w_grad * l_grad
        return total_loss, l_int, l_grad