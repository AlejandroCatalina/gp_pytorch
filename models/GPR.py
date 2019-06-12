import math
import pdb

import numpy as np
import torch
import torch.nn as nn
from torch import distributions as dist
from torch.nn.init import uniform_


class GPR(nn.Module):
    def __init__(self, kernel):
        super(GPR, self).__init__()
        self.kernel = kernel
        self.noise_std = nn.Parameter(torch.exp(uniform_(torch.empty(1, 1), -3., 0.)))
        self.X_train = None
        self.y_train = None

    def forward(self, x, y):
        self.X_train = x
        self.y_train = y
        Kxx = self.kernel(x, x) + 1e-3 * torch.eye(x.shape[0])
        nmll = -self.log_marginal_likelihood(Kxx, y)
        return nmll

    def log_marginal_likelihood(self, K, y):
        n = K.shape[0]
        L = torch.cholesky(K + self.noise_std**2 * torch.eye(n))
        a, _     = torch.solve(y.unsqueeze(0), L.unsqueeze(0))
        alpha, _ = torch.solve(a, L.unsqueeze(0))
        alpha = alpha.squeeze(0)
        return -0.5 * y.t() @ alpha \
            - torch.sum(torch.diag(L)) \
            - 0.5*n*torch.log(2*torch.tensor(math.pi))

    def predict(self, x_test, full_cov = True):
        x = self.X_train
        y = self.y_train

        N, M      = x.shape
        N_test, _ = x_test.shape

        Kxx = self.kernel(x, x) + 1e-3 * torch.eye(N)
        Kxx_inv = (Kxx + self.noise_std * torch.eye(N)).inverse()
        Ks = self.kernel(x, x_test)
        Kss = self.kernel(x_test, x_test)

        m_pred = Ks.t() @ Kxx_inv @ y
        k_pred = (Kss + self.noise_std * torch.eye(N_test) \
                  - Ks.t() @ Kxx_inv @ Ks)
        if not full_cov:
            k_pred = k_pred.diag().sqrt()
        return m_pred, k_pred
