import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class MSRSFusionDataset(Dataset):
    """
    MSRS 数据集适配器：用于 Stage 2 无监督融合训练
    改造后：忽略split参数，自动合并train和test数据用于训练
    """
    def __init__(self, root_dir, size=(384, 640), split='train'):  # 保持参数不变，兼容外部调用
        # 同时加载train和test目录的红外/可见光数据
        self.split_dirs = ['train', 'test']  # 要合并的数据集划分
        self.ir_dirs = [os.path.join(root_dir, 'Infrared', s) for s in self.split_dirs]
        self.vis_dirs = [os.path.join(root_dir, 'Visible', s) for s in self.split_dirs]
        
        # 合并所有train+test的文件名，并去重（避免同名文件重复加载）
        self.filenames = []
        for ir_dir in self.ir_dirs:
            if os.path.exists(ir_dir):  # 兼容目录不存在的情况
                dir_files = sorted([f for f in os.listdir(ir_dir) if f.endswith('.png')])
                # 记录文件名和对应的目录索引，方便后续读取
                for f in dir_files:
                    # 避免重复添加同名文件（如果train和test有重名）
                    if f not in [item[0] for item in self.filenames]:
                        self.filenames.append((f, self.ir_dirs.index(ir_dir)))
        
        # 统一 Resize 到 8 的倍数（保持原有变换逻辑）
        self.transform = T.Compose([
            T.Resize(size),
            T.ToTensor(), # 输出 [0, 1]
        ])

    def __len__(self):
        # 返回合并后的总数据量
        return len(self.filenames)

    def __getitem__(self, idx):
        # 取出文件名和对应的目录索引（train=0, test=1）
        fname, dir_idx = self.filenames[idx]
        
        # 根据目录索引读取对应split下的红外/可见光图片
        ir_path = os.path.join(self.ir_dirs[dir_idx], fname)
        vis_path = os.path.join(self.vis_dirs[dir_idx], fname)
        
        # 读取并转为灰度图（保持原有逻辑）
        ir_img = Image.open(ir_path).convert('L')
        vis_img = Image.open(vis_path).convert('L')
        
        # 数据变换（保持原有逻辑）
        ir_tensor = self.transform(ir_img)
        vis_tensor = self.transform(vis_img)
        
        # 保持返回值格式不变：ir_tensor, vis_tensor, fname
        return ir_tensor, vis_tensor, fname