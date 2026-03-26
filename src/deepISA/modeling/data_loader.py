import os
import torch
from torch.utils.data import Dataset
import json
import numpy as np



class DualDataset(Dataset):
    def __init__(self, data_dir, indices=None):
        with open(os.path.join(data_dir, "metadata.json"), "r") as f:
            meta = json.load(f)
        
        # Open memmaps in read-only mode
        self.X = np.memmap(os.path.join(data_dir, "X.npy"), dtype='float32', mode='r', shape=tuple(meta['X']))
        self.Yr = np.memmap(os.path.join(data_dir, "Yr.npy"), dtype='float32', mode='r', shape=tuple(meta['Yr']))
        self.Yc = np.memmap(os.path.join(data_dir, "Yc.npy"), dtype='float32', mode='r', shape=tuple(meta['Yc']))
        
        # If no indices provided, use the full range
        self.indices = indices if indices is not None else np.arange(meta['X'][0])
        self.len = len(self.indices)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # Map the local index to the global memmap index
        real_idx = self.indices[idx]
        
        # .copy() ensures we get a standard numpy array from the memmap 
        # before converting to tensor, which is safer for multiprocessing
        x = torch.from_numpy(self.X[real_idx].copy())
        yr = torch.tensor(self.Yr[real_idx])
        yc = torch.tensor(self.Yc[real_idx])
        return x, yr, yc