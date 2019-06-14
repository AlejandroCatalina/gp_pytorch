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
x_test_ = (x_test - x.mean()) / x.std()

kernel  = SquaredExp() # kernel
model = SGPR(D = 1, M = 20, kernel = kernel, Z = z)

def train(module, x, y_noisy, y = None, x_test = None, n_iters=50, lr = 1e-3, plot = False):
    # optimize log marginal likelihood
    opt = torch.optim.Adam(module.parameters(), lr=lr)

    # training loop
    for iter in range(n_iters):
        opt.zero_grad()
        nmll = -elbo(model, x, y_noisy)
        nmll.backward()
        opt.step()
        print(f"Iter {iter} , Log marginal likelihood : {-nmll.item()} ")
        if plot and y is not None and x_test is not None and not iter % 50:
            posterior_mean, posterior_var = model.predict(x_test, full_cov=False)
            visualize(x, y, y_noisy, x_test, posterior_mean, posterior_var, f"{model}-{iter}.pdf")

train(model, x, y, n_iters = 2500, lr = 1e-3)
posterior_mean, posterior_var = model.predict(x_test_, full_cov=False)
visualize(x, y, y_noisy, x_test, posterior_mean, posterior_var, "GP-sparse.pdf") # x , true function, noisy function, x_test, prediction_mean, pred_var, filename

#
N = 1000
X = torch.linspace(0.0, 5.0, N)
y = 0.5 * torch.sin(3*X)
y_noisy = y + dist.Normal(0.0, 0.2).sample(sample_shape=(N,))
X_ = (X - X.mean()) / X.std()
X_test = X_

