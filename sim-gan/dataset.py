import os
import torch
from torch.utils.data import Dataset
import torchvision
from PIL import Image

import torchvision.transforms as T



class SubsetDataset(Dataset):
    """
    A dataset that wraps another dataset, returning a subset from it.
    """
    def __init__(self, ds_real: Dataset, ds_sim: Dataset, subset_len):
        super().__init__()
        self.ds_real = ds_real
        self.ds_sim = ds_sim
        self.len_real = len(self.ds_real)
        self.len_sim = len(self.ds_sim)
        self.subset_len = subset_len
        

    def __getitem__(self, index):
        #  Returnrandom item
        #  Raise an IndexError if index is out of bounds.
        idx_real = torch.randint(low=0,high=self.len_real, size=(1,))
        idx_sim = torch.randint(low=0,high=self.len_sim, size=(1,))
            
        return self.ds_real[idx_real][0], self.ds_sim[idx_sim][0]
        # ========================

    def __len__(self):
        return self.subset_len
        # ========================