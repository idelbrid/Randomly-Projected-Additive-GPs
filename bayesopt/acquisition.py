import gpytorch
import numpy as np
from torch import __init__

from gp_models import AdditiveExactGPModel


def EI(xs, gp, best, use_love=False):
    """Expected improvement function (for minimization of a function)"""
    gp.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var(use_love):
        dist = gp(xs)
        mu = dist.mean
        var = dist.variance
        # diagdist = torch.distributions.Normal(mu, cov.diag())
        diagdist = torch.distributions.Normal(mu, var)
        cdf = diagdist.cdf(torch.tensor([best] * len(xs)))
        t1 = (best - mu) * cdf
        t2 = var * torch.exp(diagdist.log_prob(best))
        return t1 + t2


def ThompsonSampling(xs, gp, best, use_love=False):
    gp.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_samples(use_love):
        dist = gp(xs)
        samples = dist.sample()
    return -samples


def UCB(xs, gp, best, use_love=False):
    # Adapted from Kandasamy's Dragonfly as well.
    timestep, dim = gp.train_inputs[0].shape
    beta = np.sqrt(0.5 * dim * np.log(2 * dim * timestep + 1))
    # beta = 3
    with torch.no_grad(), gpytorch.settings.fast_pred_var(use_love):
        dist = gp(xs)
        mu = dist.mean
        var = dist.variance
    return -(mu + beta * torch.sqrt(var))


def AddUCB(xs, gp: AdditiveExactGPModel, idx):
    group = gp.get_groups()[idx]
    dim = len(group)
    timestep = len(xs)
    beta = np.sqrt(0.5 * dim * np.log(2 * dim * timestep + 1))
    with torch.no_grad():
        dist = gp.additive_pred(xs, group=idx)
        mu = dist.mean
        var = dist.variance
    return -(mu + beta * torch.sqrt(var))


def surrogate_AddUCB(xs, gp: AdditiveExactGPModel, idx):
    timestep, ambient_dim = gp.train_inputs[0].shape
    n, dim = xs.shape
    surrogate_x = torch.zeros(n, ambient_dim, dtype=torch.float)
    for i, col in enumerate(gp.get_groups()[idx]):
        surrogate_x[:, col] = xs[:, i]
    return AddUCB(surrogate_x, gp, idx)