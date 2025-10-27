import torch
from torch import nn, optim
from utils.metrics import compute_metrics
import numpy as np

def train(model, dataloader, epochs=10, lr=1e-3, device='cpu', verbose=True):
    """
    Training loop with metrics and loss curve visualization.

    Args:
        model: instance of BaseModel
        dataloader: PyTorch DataLoader
        epochs: number of epochs
        lr: learning rate
        device: 'cpu' or 'cuda'
        verbose: print metrics per epoch
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_history = []

    for epoch in range(epochs):
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(X).squeeze()       # remove singleton dimensions
            outputs = outputs.view(-1)         # flatten to 1D
            y = y.view(-1).float()             # flatten and convert to float

            loss = model.loss_fn(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X.size(0)
            all_preds.append(outputs.detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())

        # Epoch metrics
        epoch_loss = running_loss / len(dataloader.dataset)
        loss_history.append(epoch_loss)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        metrics = compute_metrics(all_labels, all_preds)

        if verbose:
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} | "
                  f"Acc: {metrics['accuracy']:.4f} | "
                  f"Precision: {metrics['precision']:.4f} | "
                  f"Recall: {metrics['recall']:.4f} | "
                  f"F1: {metrics['f1']:.4f}")

    return loss_history, metrics
