import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ABIDEDataset(Dataset):
    """
    PyTorch Dataset wrapper for ABIDE preprocessed data.
    Each subject should provide:
      - 'data': 2-channel 32x32x32 numpy array (mean & std)
      - 'label': 0 (control) or 1 (ASD)
    """
    def __init__(self, subjects, transform=None):
        self.subjects = subjects
        self.transform = transform

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        X = self.subjects[idx]['data']  # shape: 2 x 32 x 32 x 32
        y = self.subjects[idx]['label']

        if self.transform:
            X = self.transform(X)

        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def get_loader(subjects, batch_size=2, shuffle=True):
    """
    Returns a PyTorch DataLoader for the given subjects list.
    """
    dataset = ABIDEDataset(subjects)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
