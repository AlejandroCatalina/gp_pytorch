import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import uniform_


class SquaredExp(nn.Module):
    def __init__(self, D_out):
        super(SquaredExp, self).__init__()
        self.lengthscale = nn.Parameter(uniform_(torch.empty(D_out, 1, 1), -1, -0.5))
        self.prefactor = nn.Parameter(uniform_(torch.empty(D_out, 1, 1), 0.5, 1.5))

    def forward(self, X, Z):
        gamma = torch.exp(self.lengthscale)
        if Z is None:
            Z = X
        scaled_X = X / gamma ** 2
        scaled_Z = Z / gamma ** 2
        X2 = (scaled_X ** 2).sum(-1, keepdim = True)
        Z2 = (scaled_Z ** 2).sum(-1, keepdim = True)
        XZ = scaled_X.matmul(scaled_Z.transpose(1, 2))
        r2 = (X2 - 2 * XZ + Z2.transpose(1, 2)).clamp(min = 1e-6)
        return self.prefactor ** 2 * torch.exp(-0.5 * r2)
