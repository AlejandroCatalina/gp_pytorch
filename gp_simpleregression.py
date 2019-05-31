import torch
from collections import namedtuple
from models import GPR
from kernels import SquaredExp
import numpy as np

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
x = torch.linspace(-5, 5, consts.Ntrain).reshape((-1, 1, 1))
y = f(x)

y_noisy = y + np.random.normal(0, consts.noisestd) # noisy target
x_test   = torch.linspace(-10, 10, consts.Ntest).reshape((1,-1, 1)) # test data
kernel  = SquaredExp() # kernel 

model = GPR(x, y, kernel)
model.train()
posterior_mean, posterior_var = model.predict(x_test)