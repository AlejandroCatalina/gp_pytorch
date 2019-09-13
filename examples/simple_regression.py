import math
from collections import namedtuple

import numpy as np
import torch
from torch import distributions as dist

from gppytorch.kernels import SquaredExp
from gppytorch.losses import elbo
from gppytorch.models import DGP, GPR, SGPR, FlowGP
from gppytorch.utils import identity_mean
from gppytorch.visualize import visualize1d as visualize

# define the constants
consts = namedtuple("consts", "Ntrain Ntest noisestd")
consts.Ntrain = 500 # 500 training points
consts.Ntest  = 500 # 5 test points
consts.noisestd = 0.3 # noise added to data
consts.seed = 280219

torch.manual_seed(consts.seed)

# data generating function
# f = lambda x: 2*x + np.sin(5*x)
f = lambda x: 5 * np.exp(-0.5 * x**2 / 1.3**2)
# convert numpy

# load or generate training data
z = torch.linspace(-5, 5, 20).reshape((1, 20, 1))
x = torch.linspace(-5, 5, consts.Ntrain).reshape((-1, 1))
y = f(x)

# standardize input
x_ = (x - x.mean()) / x.std()

print(y.shape)
y_noisy = y + torch.randn((consts.Ntrain,1)) * consts.noisestd #np.random.normal(0, consts.noisestd, consts.Ntrain) # noisy target
x_test   = torch.linspace(-10, 10, consts.Ntest).reshape((-1, 1)) # test data
x_test_ = (x_test - x.mean()) / x.std()

kernel  = SquaredExp(D_out = 1) # kernel
model = SGPR(D_in = 1, D_out = 1, M = 20, kernel = kernel, Z = z,
             mean = lambda X: torch.mean(X, dim = 1, keepdim = True))

def train(model, x, y_noisy, y = None, x_test = None, n_iters=50, lr = 1e-3, plot = False, plot_every = 250, K = 1):
    # optimize log marginal likelihood
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # training loop
    for iter in range(n_iters):
        opt.zero_grad()
        nmll = -elbo(model, x, y_noisy, K = K)
        nmll.backward()
        opt.step()
        print(f"Iter {iter} , Log marginal likelihood : {-nmll.item()} ")
        if plot and y is not None and x_test is not None and not iter % plot_every:
            posterior_mean, posterior_var = model.predict(x_test, full_cov=False)
            visualize(x, y, y_noisy, x_test, posterior_mean, posterior_var, f"../{model}-{iter}.pdf")

train(model, x, y_noisy, y = y, x_test = x_test_, n_iters = 5000, lr = 1e-2, plot = True)
posterior_mean, posterior_var = model.predict(x_test_, full_cov=False)
visualize(x, y, y_noisy, x_test_, posterior_mean, posterior_var, "../SGPR-5000.pdf")

kernel  = SquaredExp(D_out = 1) # kernel
model = DGP(D_in = 1, layers_sizes = [3, 1], kernel = SquaredExp, M = 20)
train(model, x, y_noisy, y = y, x_test = x_test_, n_iters = 2500, lr = 1e-1, plot = True)
posterior_mean, posterior_var = model.predict(x_test_, full_cov=False)
visualize(x, y, y_noisy, x_test_, posterior_mean, posterior_var, "../DGP-2500.pdf")

iters = 0
model = FlowGP(D_in = 1, D_out = 1, T = 2.1, timestep = .3,
               kernel = SquaredExp, M = 20, mean_g = identity_mean,
               sigma_f_bounds = [1, 2], alpha_f_bounds = [0.25, 0.5],
               sigma_g_bounds = [1, 2], alpha_g_bounds = [0.25, 0.5])
increment = 100
train(model, x, y_noisy, y = y, x_test = x_test_, n_iters = increment,
      lr = 1e-2, plot = True, plot_every = 10, K = 50)
iters += increment
posterior_mean, posterior_var = model.predict(x_test_, full_cov=False)
visualize(x, y, y_noisy, x_test_, posterior_mean, posterior_var, f"../{model}-{iters}.pdf")
