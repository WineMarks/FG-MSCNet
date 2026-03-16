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
    def __init__(self, root_dir, mode='train', img_size=512):
        self.mode = mode
        self.img_size = img_size
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
        self.lmdb_path = lmdb_path
        self.env = None

        temp_env = lmdb.open(lmdb_path, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        with temp_env.begin(write=False) as txn:
            try:
                self.length = int(txn.get('num-samples'.encode('utf-8')))
            except:
                print(f"Warning: 'num-samples' key not found in {lmdb_path}. Setting length to 0.")
                self.length = 0
        temp_env.close()

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
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.OneOf([
                    A.ImageCompression(quality_lower=30, quality_upper=80, p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                ], p=0.6),

                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
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
        db_index = index + 1 
        
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % db_index
            imgbuf = txn.get(img_key.encode('utf-8'))
            
            lbl_key = 'label-%09d' % db_index
            lblbuf = txn.get(lbl_key.encode('utf-8'))

            if imgbuf is None or lblbuf is None:
                random_idx = random.randint(0, self.length - 1)
                return self.__getitem__(random_idx)
            try:
                img_np = cv2.imdecode(np.frombuffer(imgbuf, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img_np is None: raise ValueError("Image decode failed")
                img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

                mask_np = cv2.imdecode(np.frombuffer(lblbuf, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                if mask_np is None: raise ValueError("Mask decode failed")
                mask_np = np.where(mask_np > 0, 1.0, 0.0).astype(np.float32)
            except Exception as e:
                random_idx = random.randint(0, self.length - 1)
                return self.__getitem__(random_idx)
        augmented = self.transform(image=img_np, mask=mask_np)
        img_tensor = augmented['image']
        mask_tensor = augmented['mask']
        mask_tensor = mask_tensor.unsqueeze(0)

        return img_tensor, mask_tensor

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