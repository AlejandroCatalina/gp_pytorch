import torch

def log1pe(x, lower = 1e-6):
    return torch.log(1 + torch.exp(x)) + lower
