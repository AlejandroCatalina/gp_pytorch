import pdb

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import uniform_


uniform_ = torch.nn.init.uniform_

class SquaredExp(nn.Module):
    def __init__(self):
        super(SquaredExp, self).__init__()
        self.lengthscale = nn.Parameter(torch.exp(uniform_(torch.empty(1, 1), -3, 0)))
        self.prefactor = nn.Parameter(torch.exp(uniform_(torch.empty(1, 1), -1, 1)))

    def forward(self, X1, X2):
        l2      = (X1 - X2.transpose(1, 2))**2
        pref_sq = self.prefactor**2
        ls_sq   = self.lengthscale**2
        kernel  = pref_sq * torch.exp(-0.5 * l2 / ls_sq)
        return kernel
