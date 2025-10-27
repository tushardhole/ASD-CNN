import os
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class ABIDEDataset(Dataset):
    """
    Dataset for loading fMRI .nii.gz files and their labels.
    Assumes 2-channel data: mean & std or time series processed.
    """
    def __init__(self, data_dir, label_csv, transform=None):
        self.data_dir = data_dir
        self.labels_df = pd.read_csv(label_csv)
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        filename = row['filename']
        label = row['label']

        file_path = os.path.join(self.data_dir, filename)
        img = nib.load(file_path).get_fdata()  # shape: (x, y, z, time) or (x, y, z)

        # Preprocessing step — Example: Compute mean & std across time if 4D
        if img.ndim == 4:
            mean_img = np.mean(img, axis=3)
            std_img = np.std(img, axis=3)
            data = np.stack([mean_img, std_img], axis=0)  # shape: (2, x, y, z)
        else:
            data = np.expand_dims(img, axis=0)  # shape: (1, x, y, z)

        # Resize or crop to 32x32x32 if needed
        data = self._resize_or_crop(data, (32, 32, 32))

        if self.transform:
            data = self.transform(data)

        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

    def _resize_or_crop(self, data, target_shape):
        """
        Resize or crop a 3D or 4D numpy array to target_shape.
        For now, let's just center-crop.
        """
        from scipy.ndimage import zoom

        # Current shape: (channels, x, y, z)
        c, x, y, z = data.shape
        tx, ty, tz = target_shape

        # Resize using zoom if different shape
        if (x, y, z) != target_shape:
            zoom_factors = (1, tx / x, ty / y, tz / z)
            data = zoom(data, zoom_factors, order=1)

        return data

def get_loader(data_dir, label_csv, batch_size=2, shuffle=True):
    dataset = ABIDEDataset(data_dir=data_dir, label_csv=label_csv)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
