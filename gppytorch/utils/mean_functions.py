import torch

identity_mean = lambda X: torch.mean(X, dim = 1, keepdim = True)
zero_mean = lambda X: 0
