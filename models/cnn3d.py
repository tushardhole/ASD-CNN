import torch
import torch.nn as nn
from .base_model import BaseModel

class CNN3D(BaseModel):
    """
    3D CNN model similar to the 2CC3D described in the paper.
    Input: 2-channel 32x32x32 volumes (mean and std)
    """

    def __init__(self):
        super(CNN3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
