import os
import lmdb
import six
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt


class DocTamperDataset(Dataset):
    """
    针对 DocTamper LMDB 数据集的专用加载器
    修复了 Windows 下 num_workers > 0 时的 Pickle 报错问题
    """

    def __init__(self, root_dir, mode='train', img_size=512):
        self.mode = mode
        self.img_size = img_size

        # 1. 目录映射逻辑
        if mode == 'train':
            lmdb_path = os.path.join(root_dir, 'DocTamperV1-TrainingSet')
        elif mode == 'val' or mode == 'test':
            lmdb_path = os.path.join(root_dir, 'DocTamperV1-TestingSet')
        elif mode == 'fcd':
            lmdb_path = os.path.join(root_dir, 'DocTamperV1-FCD')
        elif mode == 'scd':
            lmdb_path = os.path.join(root_dir, 'DocTamperV1-SCD')
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if not os.path.exists(lmdb_path):
            raise FileNotFoundError(f"LMDB path not found: {lmdb_path}")

        # 保存路径供子进程使用
        self.lmdb_path = lmdb_path
        # 关键修改：初始化时不持有 env 对象
        self.env = None

        # 2. 临时打开一次获取长度，然后立即关闭
        temp_env = lmdb.open(lmdb_path, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        with temp_env.begin(write=False) as txn:
            try:
                self.length = int(txn.get('num-samples'.encode('utf-8')))
            except:
                print(f"Warning: 'num-samples' key not found in {lmdb_path}. Setting length to 0.")
                self.length = 0
        temp_env.close()  # 必须关闭，确保 self.env 为 None 或被销毁

        # 3. 定义数据增强
        self.transform = self._get_transforms(mode, img_size)

    # def _get_transforms(self, mode, img_size):
    #     if mode == 'train':
    #         return A.Compose([
    #             A.Resize(height=img_size, width=img_size),
    #             # A.HorizontalFlip(p=0.5),
    #             # A.VerticalFlip(p=0.5),
    #             # A.RandomRotate90(p=0.5),
    #             # A.ImageCompression(quality_lower=50, quality_upper=95, p=0.5),
    #             # A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    #             A.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225)),
    #             ToTensorV2()
    #         ])
    #     else:
    #         return A.Compose([
    #             A.Resize(height=img_size, width=img_size),
    #             A.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225)),
    #             ToTensorV2()
    #         ])
    def _get_transforms(self, mode, img_size):
        if mode == 'train':
            return A.Compose([
                A.Resize(height=img_size, width=img_size),
                
                # 1. 基础几何变换 (打破排版规律，防过拟合)
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                
                # 2. 【核心杀器】针对 FCD 的破坏性退化组合
                # 使用 OneOf 保证每次只应用一种极端破坏，防止图像彻底变成马赛克
                A.OneOf([
                    # 极限全局压缩：下限降到 40，强行抹平 SRM 的高频篡改噪声
                    A.ImageCompression(quality_lower=30, quality_upper=80, p=1.0),
                    # 缩放重采样：模拟图像在不同设备、社交软件间传输导致的模糊
                    # A.Downscale(scale_min=0.5, scale_max=0.9, p=1.0),
                    # 全局高斯模糊：抹除锐利的拼接边缘
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                ], p=0.6), # 60% 的概率遭受这种恶劣毒打
                
                # 3. 传感器噪声模拟 (对抗原图自带的底噪)
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                
                # 必须保持在最后的归一化和张量转换
                A.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(height=img_size, width=img_size),
                A.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

    def __len__(self):
        return self.length

    # def __getitem__(self, index):
    #     # 关键修改：延迟初始化 (Lazy Loading)
    #     # 每个子进程第一次调用时会创建自己的连接
    #     if self.env is None:
    #         self.env = lmdb.open(self.lmdb_path, max_readers=1, readonly=True, lock=False, readahead=False,
    #                              meminit=False)

    #     index = index + 1

    #     with self.env.begin(write=False) as txn:
    #         img_key = 'image-%09d' % index
    #         imgbuf = txn.get(img_key.encode('utf-8'))

    #         img_np = cv2.imdecode(np.frombuffer(imgbuf, dtype=np.uint8), cv2.IMREAD_COLOR)
    #         img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    #         lbl_key = 'label-%09d' % index
    #         lblbuf = txn.get(lbl_key.encode('utf-8'))

    #         mask_np = cv2.imdecode(np.frombuffer(lblbuf, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    #         mask_np = np.where(mask_np > 0, 1.0, 0.0).astype(np.float32)

    #     augmented = self.transform(image=img_np, mask=mask_np)
    #     img_tensor = augmented['image']
    #     mask_tensor = augmented['mask']
    #     mask_tensor = mask_tensor.unsqueeze(0)

    #     return img_tensor, mask_tensor
    def __getitem__(self, index):
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)

        # PyTorch 传进来的 index 是 0 到 length-1
        db_index = index + 1 
        
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % db_index
            imgbuf = txn.get(img_key.encode('utf-8'))
            
            lbl_key = 'label-%09d' % db_index
            lblbuf = txn.get(lbl_key.encode('utf-8'))

            # --- 核心防御逻辑 1：应对缺失的键值 (索引空洞) ---
            if imgbuf is None or lblbuf is None:
                # 打印一下是哪个图丢了（可选，防止刷屏可注释掉）
                # print(f"Warning: Missing data at index {db_index}, fetching a random replacement.")
                
                # 随机抽取另一个合法的 index 重新走一遍流程
                random_idx = random.randint(0, self.length - 1)
                return self.__getitem__(random_idx)
            
            # --- 核心防御逻辑 2：应对损坏的图片二进制流 ---
            try:
                img_np = cv2.imdecode(np.frombuffer(imgbuf, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img_np is None: raise ValueError("Image decode failed")
                img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

                mask_np = cv2.imdecode(np.frombuffer(lblbuf, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                if mask_np is None: raise ValueError("Mask decode failed")
                mask_np = np.where(mask_np > 0, 1.0, 0.0).astype(np.float32)
            except Exception as e:
                # print(f"Warning: Corrupted image at index {db_index}, fetching a random replacement.")
                random_idx = random.randint(0, self.length - 1)
                return self.__getitem__(random_idx)

        # 应用数据增强
        augmented = self.transform(image=img_np, mask=mask_np)
        img_tensor = augmented['image']
        mask_tensor = augmented['mask']
        mask_tensor = mask_tensor.unsqueeze(0)

        return img_tensor, mask_tensor


# --- 保持原来的辅助函数不变 ---
def get_dataloader(data_dir, mode, batch_size, img_size=512, num_workers=4):
    dataset = DocTamperDataset(root_dir=data_dir, mode=mode, img_size=img_size)
    shuffle = True if mode == 'train' else False
    drop_last = True if mode == 'train' else False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=True
    )
    return loader