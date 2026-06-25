import os
import pandas as pd
import torch
from torch.utils.data import Dataset

from .data_loading import load_data, create_samples
class radar_dataset(Dataset):
    def __init__(self, folderloc , total, list_length=10, min_list_length = 10):
        min_list_length = list_length
        
        self.data = load_data(folderloc, total)
        self.x, self.y = create_samples(self.data,  list_length=list_length, min_list_length = min_list_length)
        
        
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = self.x[idx]
        target = self.y[idx]
        
        sample = torch.stack(sample).unsqueeze(1).float()
        target = torch.tensor(target, dtype=torch.float32)
        return sample, target