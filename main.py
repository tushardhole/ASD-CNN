import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data.local_abide import get_loader
from models.cnn3d import CNN3D
from models.resnet3d import ResNet3D
from models.densenet3d import DenseNet3D
from models.r2plus1d import R2Plus1D
from data.nilearn_abide import ABIDEDataset
from training.trainer import train
from main_helpers import preprocess_fmri, fetch_abide_subjects  # helper functions

# ----------------- Config -----------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 20
batch_size = 8
n_subjects = 3  # small subset for testing

# ----------------- Load and preprocess ABIDE -----------------
data_dir = './testdata'
label_csv = './testdata/labels.csv'
loader = get_loader(data_dir, label_csv, batch_size=batch_size, shuffle=True)

# ----------------- Train and evaluate all models -----------------
print(f"\nTraining model CNN3D...")
model = CNN3D()
_, metrics = train(model, loader, epochs=epochs, lr=1e-3, device=device, verbose=True)

# ----------------- Print summary -----------------
print("\n--- Model Performance ---")
print(f"CNN3D: Acc={metrics['accuracy']:.4f}, Precision={metrics['precision']:.4f}, "
          f"Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
