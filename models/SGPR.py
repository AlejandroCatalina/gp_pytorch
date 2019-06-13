import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn.init import normal_, uniform_

from .GPR import GPR


class SGPR(GPR):
    def __init__(self, D, M, kernel, D_out = 1.):
        super(SGPR, self).__init__(kernel)
        self.M, self.D = M, D

        self.Z = nn.Parameter(normal_(torch.empty(D_out, self.M, D)))
        self.m = nn.Parameter(normal_(torch.empty(D_out, self.M, 1)))
        self.L = nn.Parameter((torch.exp(uniform_(torch.empty(D_out, M, M), -3, 0))).tril())

    def kl_multivariate_normal(self, m_q, L, m_p, S_p):
        M = self.M

        ## L is the cholesky decomposition of the covariance matrix of q
        L_inv = L.inverse()
        S_q_inv = L_inv.transpose(1, 2) @ L_inv
        S_q = L @ L.transpose(1, 2)

        log_ratio_det = torch.trace(S_p) - 2 * torch.einsum('kii', L) - M
        trace = torch.einsum('kii', S_p.inverse() @ S_q)
        mu = (m_p - m_q).transpose(1, 2) @ S_p.inverse() @ (m_p - m_q)
        return 0.5 * (log_ratio_det + trace + mu.squeeze())

    def forward(self, X, y):
        N, M = X.shape[0], self.M
        x, z = X, self.Z
        Knn = self.kernel(x, x) + (1e-3 * torch.eye(N))
        Knm = self.kernel(x, z)
        Kmm = self.kernel(z, z) + (1e-1 * torch.eye(M))
        Kmm_inv = Kmm.inverse()
        A = Knm @ Kmm_inv
        S = self.L @ self.L.transpose(1, 2)

        mu = A @ self.m
        cov = Knn + A @ (S - Kmm) @ A.transpose(1, 2)

        return mu, cov

    def KL(self):
        return self.kl_multivariate_normal(self.m, self.L,
                                           torch.zeros_like(self.M, 1),
                                           torch.eye(self.M))

    def predict(self, x_test, full_cov = True):
        mu, cov = self.forward(x_test)
        if not full_cov:
            return mu, cov.diag().sqrt()
        return mu, cov
