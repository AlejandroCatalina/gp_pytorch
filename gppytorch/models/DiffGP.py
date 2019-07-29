import torch

from torch import distributions as dist
from gppytorch.models import DGP
from gppytorch.models import SGPR

class DiffGP(DGP):
    """Implementation of differential GP according to
    Pashupati Hegde et al, Deep learning with differential Gaussian process flows in AISTATS 2019.
    """
    def __init__(self, D_in, D_out, T, timestep, kernel, M = 10):
        super(DiffGP, self).__init__(D_in, [], kernel, M)
        self.T = T
        self.M = M
        self.dt = torch.tensor(timestep)
        self.f = SGPR(D_in = D_in, M = M, kernel = kernel(D_in), D_out = D_in,
                      mean = lambda X: torch.mean(X, dim = 1, keepdim = True))
        self.g = SGPR(D_in = D_in, M = M, kernel = kernel(D_out), D_out = D_out,
                      mean = lambda X: torch.mean(X, dim = 1, keepdim = True))
        self.checkpoints = []

    def __str__(self):
        return f"DiffGP-{self.T}-{self.M}"

    def get_noise(self):
        return self.g.noise_std

    def integrate(self, X, y = None, save_checkpoints = False):
        X_t = X.t()
        for step in torch.arange(0, self.T, self.dt):
            mu_l, cov_l = self.f.forward(X_t.t(), y = y)
            mu_l = mu_l.squeeze(-1)

            eps = dist.MultivariateNormal(torch.zeros_like(mu_l),
                                          torch.eye(mu_l.shape[1])).sample()
            X_t = X_t + mu_l * self.dt + eps * self.dt.sqrt() * torch.einsum('kii->ki', cov_l).sqrt()
            if save_checkpoints:
                self.checkpoints.append(X_t)

        return X_t.t()

    def forward(self, X, y = None, save_checkpoints = True):
        """Draw one sample (trajectory) from the variational posterior."""
        X_T = self.integrate(X, y = y, save_checkpoints = save_checkpoints)
        mu, cov = self.g.forward(X_T, y)

        return mu, cov

    def KL(self):
        return torch.sum(torch.stack([node.KL().sum()
                                      for node in [self.f, self.g]]))
