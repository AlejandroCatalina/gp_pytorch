import pdb

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import uniform_


uniform_ = torch.nn.init.uniform_

class SquaredExp(nn.Module):
    def __init__(self):
        super(SquaredExp, self).__init__()
        self.lengthscale = nn.Parameter(uniform_(torch.empty(1, 1), -3, 0))
        self.prefactor = nn.Parameter(uniform_(torch.empty(1, 1), -1, 1))

    def forward(self, X, Z):
        gamma = self.lengthscale
        alpha = self.prefactor
        if Z is None:
            Z = X
        scaled_X = X / gamma ** 2
        scaled_Z = Z / gamma ** 2
        X2 = (scaled_X ** 2).sum(-1, keepdim = True)
        Z2 = (scaled_Z ** 2).sum(-1, keepdim = True)
        XZ = scaled_X.matmul(scaled_Z.transpose(1, 2))
        r2 = (X2 - 2 * XZ + Z2.transpose(1, 2)).clamp(min = 0)
        return torch.exp(alpha) ** 2 * torch.exp(gamma) ** 2 * torch.exp(-0.5 * r2)
