from sklearn.datasets import load_boston
from gppytorch.models import FlowGP, SGPR
from gppytorch.kernels import SquaredExp
from gppytorch.losses import elbo
from gppytorch.visualize import visualize1d as visualize
import torch
import random

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


def train(model, x, y_noisy, y = None, x_test = None, n_iters=50, lr = 1e-3, plot = False, K = 1):
    # optimize log marginal likelihood
    nmlls = []
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # training loop
    for iter in range(n_iters):
        opt.zero_grad()
        nmll = -elbo(model, x, y_noisy, K = K)
        nmll.backward()
        opt.step()
        nmlls.append(-nmll.item)
        print(f"Iter {iter} , Log marginal likelihood : {-nmll.item()} ")
        if plot and y is not None and x_test is not None and not iter % 250:
            posterior_mean, posterior_var = model.predict(x_test, full_cov=False)
            visualize(x, y, y_noisy, x_test, posterior_mean, posterior_var, f"../{model}-{iter}.pdf")
    return nmlls


model = FlowGP(D_in = D, D_out = 1, T = 5, timestep = .5, kernel = SquaredExp, M = 20)
train(model, X_train, y_train, n_iters = 500, lr = 1e-1)
posterior_mean, posterior_var = model.predict(X_test, full_cov=False)
print(torch.mean((y_test - posterior_mean)**2) * y_std)

sgpr = SGPR(D_in = D, D_out = 1, kernel = SquaredExp(D_out = 1), M = 20)
train(sgpr, X_train, y_train, n_iters = 1000, lr = 1e-3)
sgpr_mean, sgpr_var = sgpr.predict(X_test, full_cov=False)
print(torch.mean((y_test - sgpr_mean)**2) * y_std)
