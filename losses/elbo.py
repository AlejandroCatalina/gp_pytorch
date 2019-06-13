import torch

def elbo(model, X, y):
    nll = model.neg_log_lik(X, y)
    kl = model.KL()
    return nll - kl
