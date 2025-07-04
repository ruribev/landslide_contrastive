import os
import h5py
import torch
from torch.utils.data import Dataset

class L4SUnlabeledDataset(Dataset):
    """Dataset for unlabeled Landslide4Sense images stored in HDF5 files."""

    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.h5')])
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, self.img_files[idx])
        with h5py.File(path, 'r') as f:
            data = f[list(f.keys())[0]][()]
        img = torch.as_tensor(data, dtype=torch.float32)
        if img.shape[0] != 14 and img.shape[-1] == 14:
            img = img.permute(2, 0, 1)
        if self.transform:
            img = self.transform(img)
        return img

class L4SSegmentationDataset(Dataset):
    """Dataset for image-mask pairs from Landslide4Sense."""

    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.h5')])
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        mask_path = os.path.join(self.mask_dir, self.img_files[idx])
        with h5py.File(img_path, 'r') as f:
            img = f[list(f.keys())[0]][()]
        with h5py.File(mask_path, 'r') as f:
            mask = f[list(f.keys())[0]][()]
        img = torch.as_tensor(img, dtype=torch.float32)
        mask = torch.as_tensor(mask, dtype=torch.float32)
        if img.shape[0] != 14 and img.shape[-1] == 14:
            img = img.permute(2, 0, 1)
        if mask.ndim == 3:
            mask = mask.squeeze()
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask.unsqueeze(0)).squeeze(0)
        mask = mask.unsqueeze(0)
        return img, mask
