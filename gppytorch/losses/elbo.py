import torch

def elbo(model, X, y, K = 1):
    N, _ = X.shape
    nll = model.neg_log_lik(X, y, K = K)
    kl = model.KL()
    return nll - kl / N
