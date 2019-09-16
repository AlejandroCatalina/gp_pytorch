import torch

from torch import distributions as dist
from gppytorch.models import DGP
from gppytorch.models import SGPR
from gppytorch.kernels import SquaredExp
from gppytorch.utils import zero_mean, identity_mean


class FlowGP(DGP):
    """Implementation of differential GP according to
    Pashupati Hegde et al, Deep learning with differential Gaussian process flows in AISTATS 2019.
    """
    def __init__(self, D_in, D_out, T, timestep, kernel, M = 10,
                 mean_f = zero_mean, mean_g = zero_mean,
                 sigma_f_prior = dist.Uniform(1., 2.),
                 alpha_f_prior = dist.Uniform(0.25, 0.75),
                 sigma_g_prior = dist.Uniform(1., 2.),
                 alpha_g_prior = dist.Uniform(0.25, 0.75)):
        super(FlowGP, self).__init__(D_in, [], kernel, M)
        self.T = T
        self.M = M
        self.dt = torch.tensor(timestep).float()
        self.D_in = D_in
        self.D_out = D_out
        self.f = SGPR(D_in = D_in, M = M,
                      kernel = kernel(D_in, sigma_prior = sigma_f_prior,
                                      alpha_prior = alpha_f_prior),
                      D_out = D_in, mean = mean_f)
        self.g = SGPR(D_in = D_in, M = M,
                      kernel = kernel(D_out, sigma_prior = sigma_g_prior,
                                      alpha_prior = alpha_g_prior),
                      D_out = D_out, mean = mean_g)
        self.checkpoints = []

    def __str__(self):
        return f"FlowGP-{self.T}-{self.M}"

    def set_kernel_prior(self, sigma_f_prior, alpha_f_prior,
                         sigma_g_prior, alpha_g_prior, kernel):
        self.f.kernel = kernel(self.D_in, sigma_prior = sigma_f_prior,
                               alpha_prior = alpha_f_prior)
        self.g.kernel = kernel(self.D_out, sigma_prior = sigma_g_prior,
                               alpha_prior = alpha_g_prior)
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

    def prior_predictive_check(self, X, sigma_f_prior, alpha_f_prior,
                               sigma_g_prior, alpha_g_prior, S = 100,
                               K = 10):
        # save kernel
        f_kernel = self.f.kernel
        g_kernel = self.g.kernel

        # define new kernel
        k_f = SquaredExp(D_out = self.f.D_out, sigma_prior = sigma_f_prior,
                         alpha_prior = alpha_f_prior)
        k_g = SquaredExp(D_out = self.f.D_out, sigma_prior = sigma_g_prior,
                         alpha_prior = alpha_g_prior)
        self.f.kernel = k_f
        self.g.kernel = k_g
        predictions = [torch.stack([self.forward(X)[0] # save only mu
                                    for _ in range(K)]).reshape(K, -1, self.g.D_out)
                       for _ in range(S)]

        # restore kernel
        self.f.kernel = f_kernel
        self.g.kernel = g_kernel
        return predictions
