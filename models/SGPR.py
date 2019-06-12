import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import normal_, uniform_

from .GPR import GPR


class SGPR(GPR):
    def __init__(self, X, y, kernel, M = 100):
        super(SGPR, self).__init__(X, y, kernel)
        self.M = M
        self.N, D = X.shape
        self.Z = nn.Parameter(normal_(torch.empty(M, D)))
        self.m = nn.Parameter(normal_(torch.empty(M, D)))
        self.S = nn.Parameter(torch.exp(uniform_(torch.empty(1), -3, 0)) * torch.eye(M))

    def forward(self):
        N, M = self.N, self.M
        x, y, z = self.X, self.y, self.Z
        Knn = self.kernel(x, x) + (1e-3 * torch.eye(N))
        Knm = self.kernel(x, z)
        Kmm = self.kernel(z, z) + (1e-1 * torch.eye(M))
        Qnn = Knm @ Kmm.inverse() @ Knm.t()
        elbo = -self.elbo(Knn, Knm, Kmm, Qnn, y)
        return elbo

    def kl(self, m_q, S_q, m_p, S_p):
        M = self.M
        log_ratio_det = torch.log(S_p.det() / S_q.det()) - M
        trace = torch.trace(S_p.inverse() @ S_q)
        mu = (m_p - m_q).t() @ S_p.inverse() @ (m_p - m_q)
        return 0.5 * (log_ratio_det + trace + mu)

    def elbo(self, Knn, Knm, Kmm, Qnn, y):
        """Implement variational ELBO as in
        Hensman et al, Scalable Variational Gaussian Process Classification, ICML 2015.
        """
        N = self.N
        Kmm_inv = Kmm.inverse()
        Knm_Kmm_inv = Knm @ Kmm_inv
        m = Knm_Kmm_inv @ self.m
        S = self.noise_std**2 * torch.eye(N)

        ## compute KL[q_u || p_u], with p_u = N(0, I) and q_u = N(m, S)
        kl_q_u_p_u = self.kl(self.m, self.S, torch.zeros_like(self.m), torch.eye(self.M))

        ## return elbo
        ## TODO: rewrite with cholesky decomposition
        return - 0.5 * (y - m).t() @ S.inverse() @ (y - m) \
            - 1. / (2 * self.noise_std**2) \
            * torch.sum(torch.diag(Knm_Kmm_inv @ self.S @ Kmm_inv @ Knm.t())) \
            - 1. / (2 * self.noise_std**2) * torch.sum(torch.diag(Knn - Qnn)) \
            - 0.5 * N * torch.log(2 * torch.tensor(math.pi)) \
            - kl_q_u_p_u

    def predict(self, x_test):
        raise NotImplementedError
