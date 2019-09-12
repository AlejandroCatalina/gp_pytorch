import math

import torch
import torch.nn as nn
from gppytorch.models import SGPR
from torch import distributions as dist


class DGP(nn.Module):
    def __init__(self, D_in, layers_sizes, kernel, M = 10):
        super(DGP, self).__init__()
        self.layers = nn.ModuleList([])
        for D_out in layers_sizes:
            self.layers.append(SGPR(D_in, M = M, kernel = kernel(D_out), D_out = D_out,
                                    mean = lambda X: torch.mean(X, dim = 1, keepdim = True)))
            D_in = D_out

    def __str__(self):
        layers_sizes = [str(self.layers[0].D_in)]
        layers_sizes.extend([str(node.D_out) for node in self.layers])
        sizes_str = "-".join(layers_sizes)
        return f"DGP-{sizes_str}"

    def forward(self, X, y = None):
        f_l = X.t()
        for node in self.layers:
            mu_l, cov_l = node.forward(f_l.t(), y = y)
            mu_l = mu_l.squeeze(-1)

            eps = dist.MultivariateNormal(torch.zeros_like(mu_l),
                                          torch.eye(mu_l.shape[1])).sample()
            f_l = mu_l + eps * torch.einsum('kii->ki', cov_l).sqrt()

        return f_l, cov_l

    def get_noise(self):
        return self.layers[-1].noise_std

    def neg_log_lik(self, X, y, K = 1):
        N = X.shape[0]

        # draw K samples from the DGP posterior approximation
        f_l, cov_l = [], []
        for _ in range(K):
            f_hat_l, cov_hat_l = self.forward(X, y)
            f_l.append(f_hat_l.squeeze(-1).t())
            cov_l.append(cov_hat_l)

        # f_L has shape (K, N, 1), cov_L has shape (K, N, N)
        f_L = torch.stack(f_l)
        cov_L = torch.stack(cov_l).squeeze(1)

        noise_std = self.get_noise()
        S = noise_std ** 2 * torch.eye(N)
        return (- 0.5 * ((y - f_L).transpose(1, 2) @ S.inverse() @ (y - f_L)).squeeze()
                - N / (2 * noise_std ** 2) * torch.einsum('kii', cov_L)
                - 0.5 * N * torch.log(2 * torch.tensor(math.pi))).mean()

    def KL(self):
        return torch.sum(torch.stack([node.KL().sum() for node in self.layers]))

    def predict(self, x_test, full_cov = True, K = 100):
        # draw K samples from the DGP posterior approximation
        f_l, cov_l = [], []
        for _ in range(K):
            f_hat_l, cov_hat_l = self.forward(x_test, y = None)
            f_l.append(f_hat_l.squeeze(-1).t())
            cov_l.append(cov_hat_l)

        # f_L has shape (K, N, 1), cov_L has shape (K, N, N)
        f_L = torch.stack(f_l)
        cov_L = torch.stack(cov_l).squeeze()

        mu = torch.mean(f_L, dim = 0)
        cov = torch.mean(cov_L, dim = 0)

        if not full_cov:
            var = cov.diag().sqrt()
            return mu, var
        return mu, cov
