import numpy as np
import torch
import torch.nn as nn
from torch import distributions as dist
from torch.nn.init import uniform_
from gppytorch.utils.transforms import log1pe

class SquaredExp(nn.Module):
    def __init__(self, D_out, sigma_prior = dist.Uniform(1., 2.),
                 alpha_prior = dist.Uniform(1., 2.)):
        super(SquaredExp, self).__init__()
        self.sigma = nn.Parameter(sigma_prior.sample(sample_shape = (1, 1)))
        self.alpha = nn.Parameter(alpha_prior.sample(sample_shape = (1, 1)))

    def forward(self, X, Z):
        gamma = log1pe(self.sigma)
        if Z is None:
            Z = X
        scaled_X = X / gamma ** 2
        scaled_Z = Z / gamma ** 2
        X2 = (scaled_X ** 2).sum(-1, keepdim = True)
        Z2 = (scaled_Z ** 2).sum(-1, keepdim = True)
        XZ = scaled_X.matmul(scaled_Z.transpose(1, 2))
        r2 = (X2 - 2 * XZ + Z2.transpose(1, 2)).clamp(min = 1e-6)
        return log1pe(self.alpha) ** 2 * torch.exp(-0.5 * r2)
