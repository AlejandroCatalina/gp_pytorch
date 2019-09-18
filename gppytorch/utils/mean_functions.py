import torch

def identity_mean(X):
    return torch.mean(X, dim = 1, keepdim = True)

def zero_mean(X):
    return 0
