import math
import pdb

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import uniform_, normal_


class GPR(nn.Module):
    def __init__(self, kernel):
        super(GPR, self).__init__()
        self.kernel = kernel
        self.noise_std = nn.Parameter(torch.exp(uniform_(torch.empty(1, 1), -3., 0.)))

    def forward(self, X, y):
        N, _ = X.shape

        Kxx = self.kernel(X, X) + 1e-3 * torch.eye(N)
        Kxx_inv = (Kxx + self.noise_std * torch.eye(N)).inverse()

        self.Kxx_inv = Kxx_inv
        self.X = X

        mu = Kxx.t() @ Kxx_inv @ y
        cov = (Kxx + self.noise_std * torch.eye(N)
               - Kxx.t() @ Kxx_inv @ Kxx)
        return mu, cov

    def neg_log_lik(self, X, y):
        mu, cov = self.forward(X, y)
        S = self.noise_std ** 2 * torch.eye(self.N)

        return (- 0.5 * (y - mu).t() @ S.inverse() @ (y - mu)
                - 1. / (2 * self.noise_std ** 2) * torch.trace(cov)
                - 0.5 * N * torch.log(2 * torch.tensor(math.pi)))

    def KL(self):
        return 0.

    def predict(self, X_test, full_cov = True):
        N = X_test.shape[0]
        X, Kxx_inv = self.X, self.Kxx_inv

        Ks = self.kernel(X_test, X) + 1e-3 * torch.eye(N)
        Kss = self.kernel(X_test, X_test) + 1e-3 * torch.eye(N)

        mu = Ks.t() @ Kxx_inv @ y
        cov = (Ks + self.noise_std * torch.eye(N)
               - Ks.t() @ Kxx_inv @ Ks)
        if not full_cov:
            return mu, cov.diag().sqrt()
        return mu, cov
