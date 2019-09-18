import torch

def log1pe(x):
    return torch.log(1 + torch.exp(x))
