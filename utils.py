import torch

def masked_mse(pred, target, mask):
    mask = mask.float()
    return ((pred - target) ** 2 * mask).sum() / mask.sum().clamp(min=1)
import torch

def masked_rmse(pred, target, mask):
    mask = mask.float()
    mse = ((pred - target) ** 2 * mask).sum() / mask.sum().clamp(min=1)
    return torch.sqrt(mse)
