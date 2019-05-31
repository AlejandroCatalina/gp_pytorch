import torch
import torch.nn as nn
import numpy as np
import pdb

class SquaredExp(nn.Module):
    def __init__(self):
        super(SquaredExp, self).__init__()
        self.lengthscale = nn.Parameter(torch.nn.init.normal_(torch.empty(1,1)))
        self.prefactor = nn.Parameter(torch.nn.init.normal_(torch.empty(1,1)))
        # print(f"lengthscale = {self.lengthscale.shape}")
        # self.pdist = nn.PairwiseDistance(p=2)

    def forward(self, X1, X2):
        l2      = (X1 - X2.transpose(-1, 0))**2
        # print(f"l2 {l2.shape}")
        pref_sq = self.prefactor**2
        ls_sq   = self.lengthscale**2
        # print(f"pref_sq {pref_sq.shape}") 
        # print(f"ls_sq {ls_sq.shape}") 
        # so that the following operations can be broadcast.
        l2      = l2.transpose(1, -1) 
        kernel  = pref_sq * torch.exp(-0.5 * l2 / ls_sq)
        return kernel.squeeze()