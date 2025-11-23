import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from pathlib import Path
import os

class CasiaBDataset(Dataset):
    def __init__(self, data_root, skeleton_root, transform=None, seq_len=30, mode='train'):
        """
        CASIA-B 数据集加载器
        
        Args:
            data_root (str): 轮廓图/RGB图根目录
            skeleton_root (str): 骨架数据根目录 (.npy files)
            transform (callable, optional): 图像预处理
            seq_len (int): 采样序列长度
            mode (str): 'train' or 'test'
        """
        self.data_root = Path(data_root)
        self.skeleton_root = Path(skeleton_root)
        self.transform = transform
        self.seq_len = seq_len
        self.mode = mode
        
        self.data_list = self._build_data_list()
        
    def _build_data_list(self):
        data_list = []
        # 简单划分: 前74个ID为训练集，后50个为测试集 (CASIA-B 标准协议之一)
        all_subjects = sorted([d.name for d in self.data_root.iterdir() if d.is_dir()])
        
        if self.mode == 'train':
            subjects = all_subjects[:74]
        else:
            subjects = all_subjects[74:]
            
        for sub in subjects:
            sub_path = self.data_root / sub
            skel_sub_path = self.skeleton_root / sub
            
            if not sub_path.exists() or not skel_sub_path.exists():
                continue
                
            for status in sorted([d.name for d in sub_path.iterdir() if d.is_dir()]):
                status_path = sub_path / status
                skel_status_path = skel_sub_path / status
                
                for view in sorted([d.name for d in status_path.iterdir() if d.is_dir()]):
                    # 检查是否有对应的骨架文件
                    skel_file = skel_status_path / view / 'skeletons.npy'
                    if not skel_file.exists():
                        continue
                        
                    # 获取图像列表
                    img_dir = status_path / view
                    imgs = sorted(list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg')))
                    if len(imgs) < 5: # 忽略过短序列
                        continue
                        
                    data_list.append({
                        'img_paths': [str(p) for p in imgs],
                        'skel_path': str(skel_file),
                        'label': int(sub) - 1, # 假设 ID 是数字文件夹名
                        'type': status,
                        'view': view
                    })
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # 1. 加载骨架
        skeletons = np.load(item['skel_path']) # (T_orig, 17, 3)
        
        # 2. 加载图像
        img_paths = item['img_paths']
        
        # 3. 采样 (固定长度)
        total_frames = len(img_paths)
        if total_frames < self.seq_len:
            # 补齐
            indices = np.arange(total_frames)
            indices = np.concatenate([indices, np.random.choice(indices, self.seq_len - total_frames)])
            indices = np.sort(indices)
        else:
            # 随机采样或均匀采样
            indices = np.linspace(0, total_frames - 1, self.seq_len).astype(int)
            
        selected_imgs = []
        for i in indices:
            img = cv2.imread(img_paths[i], cv2.IMREAD_GRAYSCALE) # 假设是轮廓图
            if img is None:
                # 简单的错误处理，生成黑图
                img = np.zeros((64, 64), dtype=np.uint8)
            else:
                img = cv2.resize(img, (64, 64))
            selected_imgs.append(img)
            
        imgs_tensor = np.array(selected_imgs) # (T, H, W)
        imgs_tensor = imgs_tensor[:, np.newaxis, :, :] # (T, C, H, W), C=1
        imgs_tensor = torch.from_numpy(imgs_tensor).float() / 255.0
        
        # 处理骨架采样
        # 注意：骨架帧数可能与图像不完全一致（如果提取时有丢帧），这里假设是一一对应的
        # 如果骨架长度不够，也需要处理
        if len(skeletons) > 0:
            if len(skeletons) != total_frames:
                 # 简单对齐：缩放索引
                 skel_indices = np.linspace(0, len(skeletons)-1, self.seq_len).astype(int)
            else:
                 skel_indices = indices
            
            selected_skels = skeletons[skel_indices] # (T, 17, 3)
        else:
            selected_skels = np.zeros((self.seq_len, 17, 3))

        skels_tensor = torch.from_numpy(selected_skels).float()
        
        label = torch.tensor(item['label'], dtype=torch.long)
        
        return imgs_tensor, skels_tensor, label
