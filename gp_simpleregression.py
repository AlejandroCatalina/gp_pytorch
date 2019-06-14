import torch
from collections import namedtuple
from models import GPR, SGPR
from losses import elbo
from kernels import SquaredExp
import numpy as np
from visualize import visualize1d as visualize

# define the constants
consts = namedtuple("consts", "Ntrain Ntest noisestd")
consts.Ntrain = 500 # 500 training points
consts.Ntest  = 500 # 5 test points
consts.noisestd = 0.3 # noise added to data


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
kernel  = SquaredExp() # kernel

model = SGPR(D = 1, M = 20, kernel = kernel, Z = z)

def train(module, x, y, n_iters=50, lr = 1e-3):
    # optimize log marginal likelihood
    opt = torch.optim.Adam(module.parameters(), lr=lr)

    # training loop
    for iter in range(n_iters):
        opt.zero_grad()
        nmll = -elbo(model, x, y)
        nmll.backward()
        opt.step()
        print(f"Iter {iter} , Log marginal likelihood : {-nmll.item()} ")
        print(f"Kernel lengthscale {module.kernel.lengthscale.item()}")
        print(f"Kernel prefactor {module.kernel.prefactor.item()}")
        print(f"Noise std {module.noise_std.item()}")

train(model, x, y, n_iters = 2500, lr = 1e-3)
posterior_mean, posterior_var = model.predict(x_test, full_cov=False)
visualize(x, y, y_noisy, x_test, posterior_mean, posterior_var, "GP-sparse.pdf") # x , true function, noisy function, x_test, prediction_mean, pred_var, filename
