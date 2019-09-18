import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import normal_, uniform_

from gppytorch.kernels import SquaredExp
from gppytorch.utils.transforms import log1pe


class GPR(nn.Module):
    def __init__(self, D_out = 1, kernel = None):
        super(GPR, self).__init__()
        self.kernel = kernel
        self.D_out = D_out
        self.noise_std = nn.Parameter(torch.tensor(1.))

    def forward(self, X, y):
        N, _ = X.shape

        N_noise = (1e-3 * torch.stack([torch.eye(N) for _ in range(self.D_out)]))
        Kxx = self.kernel(X.unsqueeze(0).repeat(self.D_out, 1, 1),
                          X.unsqueeze(0).repeat(self.D_out, 1, 1)) + N_noise
        Kxx_inv = (Kxx + N_noise).inverse()

        self.Kxx_inv = Kxx_inv
        self.X = X

        mu = Kxx.transpose(1, 2) @ Kxx_inv @ y
        cov = (Kxx + N_noise - Kxx.transpose(1, 2) @ Kxx_inv @ Kxx)
        return mu, cov

    def neg_log_lik(self, X, y, K = None):
        N = X.shape[0]
        mu, cov = self.forward(X, y)
        noise = log1pe(self.noise_std)
        S = noise_std ** 2 * torch.eye(N)

        return (- 0.5 * ( (y - mu).transpose(1, 2) @ S.inverse() @ (y - mu) ).squeeze()
                - N / (2 * noise_std ** 2) * torch.einsum('kii', cov).squeeze()
                - 0.5 * N * torch.log(2 * torch.tensor(math.pi)))

    def KL(self):
        return 0.

    def predict(self, X_test, full_cov = True):
        N = X_test.shape[0]
        X, Kxx_inv = self.X, self.Kxx_inv

        Ks = self.kernel(X_test.unsqueeze(0).repeat(self.D_out, 1, 1),
                         X.unsqueeze(0).repeat(self.D_out, 1, 1)) + 1e-3 * torch.eye(N)
        Kss = self.kernel(X_test.unsqueeze(0).repeat(self.D_out, 1, 1),
                          X_test.unsqueeze(0).repeat(self.D_out, 1, 1)) + 1e-3 * torch.eye(N)

        mu = ( Ks.transpose(1, 2) @ Kxx_inv @ y ).squeeze().reshape(-1, 1)
        cov = (Ks + self.noise_std * torch.eye(N)
               - Ks.transpose(1, 2) @ Kxx_inv @ Ks).squeeze()
        if not full_cov:
            return mu, cov.diag().sqrt()
        return mu, cov

    def prior_predictive_check(self, X, sigma_bounds, alpha_bounds, S = 100, K = 10):
        # save kernel
        kernel = self.kernel

        # define new kernel
        sigma_lower, sigma_upper = sigma_bounds
        alpha_lower, alpha_upper = alpha_bounds
        k = SquaredExp(D_out = self.D_out, sigma_lower_bound = sigma_lower,
                       sigma_upper_bound = sigma_upper, alpha_lower_bound = alpha_lower,
                       alpha_upper_bound = alpha_upper)
        self.kernel = k
        predictions = [torch.stack([self.forward(X) for _ in range(K)]) for _ in range(S)]

        # restore kernel
        self.kernel = kernel
        return predictions
