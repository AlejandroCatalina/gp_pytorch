import random

import numpy as np
import torch
from sklearn.datasets import load_boston
from torch import distributions as dist

from gppytorch.kernels import SquaredExp
from gppytorch.losses import elbo
from gppytorch.models import SGPR, FlowGP
from gppytorch.utils.mean_functions import identity_mean
from gppytorch.visualize import visualize1d as visualize

torch.manual_seed(280219)
boston = load_boston()
X = torch.from_numpy(boston['data']).float()
X = (X - X.mean()) / X.std()
y = torch.from_numpy(boston['target']).reshape(-1, 1).float()
y_mean = y.mean()
y_std = y.std()
y = (y - y.mean()) / y.std()

N, D = X.shape
train_size = int(0.9 * N)
train_inds = random.sample(list(np.arange(0, N)),  train_size)
test_inds = np.setdiff1d(np.arange(0, N), train_inds)

X_train, X_test = X[train_inds, :], X[test_inds, :]
y_train, y_test = y[train_inds, :], y[test_inds, :]


def train(model, x, y_noisy, y = None, x_test = None, y_test = None, n_iters=50, lr = 1e-3,
          plot = False, plot_every = 50, print_every = 50, K = 1):
    if x_test is None:
        x_test = x
    if y is None:
        y = y_noisy
    if y_test is None:
        y_test = y
    # optimize log marginal likelihood
    nmlls = []
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # training loop
    for iter in range(n_iters):
        opt.zero_grad()
        nmll = -elbo(model, x, y_noisy, K = K)
        nmll.backward()
        opt.step()
        nmlls.append(-nmll.item())
        if iter % print_every == 0:
            posterior_mean, posterior_var = model.predict(x_test, full_cov=False)
            test_rmse = ((y_test - posterior_mean)**2).mean().sqrt()
            print(f"Iter {iter}, log marginal likelihood: {-nmll.item()}, rmse: {test_rmse}")
        if plot and y is not None and x_test is not None and iter % plot_every == 0:
            posterior_mean, posterior_var = model.predict(x_test, full_cov=False)
            visualize(x, y, y_noisy, x_test, posterior_mean, posterior_var, f"../{model}-{iter}.pdf")
    return nmlls


sigma_prior = dist.Uniform(1, 2)
alpha_prior = dist.Uniform(1, 2)
model = FlowGP(D_in = 1, D_out = 1, T = 2.1, timestep = .3,
               kernel = SquaredExp, M = 100, sigma_f_prior = sigma_prior,
               alpha_f_prior = alpha_prior, sigma_g_prior = sigma_prior,
               alpha_g_prior = alpha_prior, mean_f = identity_mean,
               mean_g = identity_mean)

train(model, X_train, y_train, n_iters = 5000, lr = 1e-2, K = 5)
posterior_mean, posterior_var = model.predict(X_test, full_cov=False)
print(torch.mean((y_test - posterior_mean)**2) * y_std)

sgpr = SGPR(D_in = D, D_out = 1, kernel = SquaredExp(D_out = 1), M = 20)
train(sgpr, X_train, y_train, n_iters = 1000, lr = 1e-3)
sgpr_mean, sgpr_var = sgpr.predict(X_test, full_cov=False)
print(torch.mean((y_test - sgpr_mean)**2) * y_std)
