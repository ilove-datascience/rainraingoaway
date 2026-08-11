import os
import pandas as pd
import torch
from torch.utils.data import Dataset

from .data_loading import load_data_multimodal, create_samples

class radar_dataset_multimodal(Dataset):
    def __init__(self, folderloc_radar, folderloc_env , total, list_length=10, min_list_length = 10, num_workers=None):
        min_list_length = list_length
        
        self.data = load_data_multimodal(
            folder_path_radar=folderloc_radar,
            folder_path_env=folderloc_env,
            total=total,
            num_workers=num_workers,
        )
        self.x, self.y = create_samples(self.data,  list_length=list_length, min_list_length = min_list_length)
        
        
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = self.x[idx]
        target = self.y[idx]

        # sample: [T, C, H, W] where C=7 (radar + env channels)
        sample = torch.stack(sample).float()

        # Predict next radar frame only (channel 0), shape [H, W]
        target = target[0].float()
        return sample, target