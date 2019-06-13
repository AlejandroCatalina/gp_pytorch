import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn.init import normal_, uniform_

from .GPR import GPR


class SGPR(GPR):
    def __init__(self, X, y, Z, kernel):
        super(SGPR, self).__init__(X, y, kernel)
        self.M, d = Z.shape
        self.N, D = X.shape
        assert d == D, "Dimension of Z must be equal to dimension of X"

        self.Z = nn.Parameter(Z)
        self.m = nn.Parameter(normal_(torch.empty(self.M, D)))
        self.L = nn.Parameter((torch.exp(uniform_(torch.empty(1), -3, 0)) * torch.eye(self.M)).tril())

    def forward(self):
        N, M = self.N, self.M
        x, y, z = self.X, self.y, self.Z
        Knn = self.kernel(x, x) + (1e-3 * torch.eye(N))
        Knm = self.kernel(x, z)
        Kmm = self.kernel(z, z) + (1e-1 * torch.eye(M))
        Qnn = Knm @ Kmm.inverse() @ Knm.t()
        elbo = -self.elbo(Knn, Knm, Kmm, Qnn, y)
        return elbo

    def kl(self, m_q, L, m_p, S_p):
        M = self.M

        ## L is the cholesky decomposition of the covariance matrix of q
        L_inv = L.inverse()
        S_q_inv = L_inv.t() @ L_inv
        S_q = L @ L.t()

        log_ratio_det = torch.trace(S_p) - 2 * torch.trace(L) - M
        trace = torch.trace(S_p.inverse() @ S_q)
        mu = (m_p - m_q).t() @ S_p.inverse() @ (m_p - m_q)
        return 0.5 * (log_ratio_det + trace + mu)

    def elbo(self, Knn, Knm, Kmm, Qnn, y):
        """Implement variational ELBO as in
        J. Hensman, N. Fusi, and N. D. Lawrence. Gaussian processes for big data.
          In A. Nicholson and P. Smyth, editors, Uncertainty in Artificial Intelligence, volume 29. AUAI Press, 2013.
        J. Hensman, A. G.G. Mathews and Z. Ghahramani. Scalable Variational Gaussian Process Classification.
          In Proceedings of the 18th International Conference on Artificial Intelligence and Statistics (AISTATS) 2015,
          San Diego, CA, USA. JMLR: W&CP volume 38.
        """
        N = self.N
        Kmm_inv = Kmm.inverse()
        Knm_Kmm_inv = Knm @ Kmm_inv

        m = Knm_Kmm_inv @ self.m
        S = self.noise_std**2 * torch.eye(N)

        LL = self.L @ self.L.t()

        ## compute KL[q_u || p_u], with p_u = N(0, I) and q_u = N(m, S)
        kl_q_u_p_u = self.kl(self.m, self.L, torch.zeros_like(self.m), torch.eye(self.M))

        ## return elbo
        ## TODO: rewrite with cholesky decomposition
        return - 0.5 * (y - m).t() @ S.inverse() @ (y - m) \
            - 1. / (2 * self.noise_std**2) \
            * torch.trace(Knm_Kmm_inv @ LL @ Kmm_inv @ Knm.t()) \
            - 1. / (2 * self.noise_std**2) * torch.trace(Knn - Qnn) \
            - 0.5 * N * torch.log(2 * torch.tensor(math.pi)) \
            - kl_q_u_p_u

    def predict(self, x_test, full_cov = True):
        Ntest, D = x_test.shape
        M = self.M
        x, y, z = self.X, self.y, self.Z
        Kss = self.kernel(x_test, x_test) + (1e-3 * torch.eye(Ntest))
        Ksm = self.kernel(x_test, z)
        Kmm = self.kernel(z, z) + (1e-1 * torch.eye(M))
        Kmm_inv = Kmm.inverse()
        S = self.L @ self.L.t()

        As = Ksm @ Kmm_inv

        mu = As @ self.m
        cov = Kss + As @ (S - Kmm) @ As.t() + self.noise_std ** 2 * torch.eye(Ntest)

        if not full_cov:
            cov = torch.diag(cov).sqrt()

        return mu, cov
