import torch
import torch.nn as nn

class BaseModel(nn.Module):
    """
    Base model interface. All models should inherit from this.
    Provides a standard loss function and forward interface.
    """
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, x):
        raise NotImplementedError("Forward method must be implemented by subclass.")

    def loss_fn(self, outputs, targets):
        """
        Binary classification loss.
        """
        return nn.BCELoss()(outputs, targets)
