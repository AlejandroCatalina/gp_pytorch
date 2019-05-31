import torch
import torch.nn as nn
import numpy as np
import math


class GPR(object):
    def __init__(self, xtrain, ytrain, kernel):
        self.x = xtrain
        self.y = ytrain
        self.kernel_fn = kernel
        self.noise_std = nn.Parameter(torch.nn.init.normal_(torch.empty(1,1)))
    
    def log_marginal_likelihood(self, K, y):
        n = K.shape[0]
        L = torch.cholesky(K + self.noise_std**2 * torch.eye(n))
        print(f"L {L.shape}")
        print(f"y {self.y.shape}")
        a, _      = torch.solve(self.y.squeeze(-1), L.transpose(-1, 0))
        alpha, _  = torch.solve(a, L.transpose(-1, 0))
        print(f"alpha {alpha.shape}")
        print(f"L {L.shape}")
        log_m_lik = list([0])*3
        log_m_lik[0] = self.y.transpose(-1, 0) @ alpha
        log_m_lik[0] = -0.5 * log_m_lik[0]
        log_m_lik[1] = - torch.sum(torch.diag(L.squeeze())) 
        log_m_lik[2] = - 0.5*n*torch.log(2*torch.tensor(math.pi))
        return torch.sum(torch.tensor(log_m_lik)) # -0.5 * self.y.transpose(-1, 0) @ alpha - torch.sum(torch.diag(L)) - 0.5*n*torch.log(2*np.pi)

    def train(self, n_iters=50):
        self.Kxx = self.kernel_fn(self.x, self.x)
        print(f"Kxx = {self.Kxx.shape}")

        # optimize log marginal likelihood
        log_marginal_lik = self.log_marginal_likelihood(self.Kxx, self.y)

    def predict(self, x):
        raise NotImplementedError