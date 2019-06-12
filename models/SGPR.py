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

    def forward(self):
        N, M = self.N, self.M
        x, y, z = self.X, self.y, self.Z
        Knn = self.kernel(x, x) + (1e-3 * torch.eye(N))
        Knm = self.kernel(x, z)
        Kmm = self.kernel(z, z) + (1e-1 * torch.eye(M))
        Qnn = Knm @ Kmm.inverse() @ Knm.t()
        elbo = -self.elbo(Knn, Knm, Kmm, Qnn, y)
        return elbo

    def elbo(self, Knn, Knm, Kmm, Qnn, y):
        N = self.N
        K = Knm @ Kmm.inverse() @ Knm.t() + 1e-1 * torch.eye(N)
        L = torch.cholesky(K + self.noise_std**2 * torch.eye(N))
        a, _      = torch.solve(y, L.transpose(-1, 0))
        alpha, _  = torch.solve(a, L.transpose(-1, 0))
        return - 0.5 * y.t() @ alpha \
            - 1. / (2 * self.noise_std**2) * torch.sum(torch.diag(Knn - Qnn)) \
            - 0.5 * N * torch.log(2 * torch.tensor(math.pi))

    def predict(self, x_test):
        raise NotImplementedError
