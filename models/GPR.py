import torch
import torch.nn as nn
import numpy as np
import math
import pdb

class GPR(object):
    def __init__(self, xtrain, ytrain, kernel):
        self.x = xtrain
        self.y = ytrain
        self.kernel_fn = kernel
        # self.noise_std = nn.Parameter(torch.nn.init.normal_(torch.empty(1,1)))
        self.noise_std = nn.Parameter(torch.exp(torch.nn.init.uniform_(torch.empty(1,1), -3., 0.)))
        # self.noise_std = nn.Parameter(0.01+(torch.empty(1,1)))
    
    def log_marginal_likelihood(self, K, y):
        n = K.shape[0]
        L = torch.cholesky(K + self.noise_std**2 * torch.eye(n))
        # print(f"L {L.shape}")
        # print(f"y {self.y.shape}")
        a, _      = torch.solve(self.y.squeeze(-1), L.transpose(-1, 0))
        alpha, _  = torch.solve(a, L.transpose(-1, 0))
        # print(f"alpha {alpha.shape}")
        # print(f"L {L.shape}")
        # log_m_lik = list([0])*3
        # log_m_lik[0] = self.y.transpose(-1, 0) @ alpha
        # log_m_lik[0] = -0.5 * log_m_lik[0]
        # log_m_lik[1] = - torch.sum(torch.diag(L.squeeze())) 
        # log_m_lik[2] = - 0.5*n*torch.log(2*torch.tensor(math.pi))
        # return torch.sum(torch.tensor(log_m_lik)) # -0.5 * self.y.transpose(-1, 0) @ alpha - torch.sum(torch.diag(L)) - 0.5*n*torch.log(2*np.pi)
        return -0.5 * self.y.transpose(-1, 0) @ alpha \
               - torch.sum(torch.diag(L.squeeze())) \
               - 0.5*n*torch.log(2*torch.tensor(math.pi))

    def train(self, n_iters=1000):
        # optimize log marginal likelihood
        parameters = [{'params' :  self.kernel_fn.parameters(),
                       'params' :  self.noise_std}
                    ]
        opt = torch.optim.Adam(parameters, lr=1e-4)
        
        # training loop
        for iter in range(n_iters):
            opt.zero_grad()
            self.Kxx = self.kernel_fn(self.x, self.x)
            # print(f"Kxx = {self.Kxx.shape}")
            log_marginal_lik = -1 * self.log_marginal_likelihood(self.Kxx, self.y)
            log_marginal_lik.backward(retain_graph=True)
            opt.step()
            print(f"Iter {iter} , Log marginal likelihood : {-1*log_marginal_lik} ")
            print(f"Kernel lengthscale {self.kernel_fn.lengthscale}")
            print(f"Kernel prefactor {self.kernel_fn.prefactor}")
            print(f"Noise std {self.noise_std}")

    def predict(self, x):
        raise NotImplementedError