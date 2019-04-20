import torch
import gpytorch
from math import pi
from typing import Callable, Iterable
# from scipydirect import minimize
from scipy.optimize import differential_evolution
import gp_models
import numpy as np

def sybtang(x: torch.Tensor):
    return 1/2 * torch.sum(x.pow(4) - 16*x.pow(2) + 5*x, dim=-1)


def michalewicz(x: torch.Tensor, m: int):
    d = x.shape[-1]
    return -torch.sum(torch.sin(x) * torch.sin(d * x / pi).pow(2*m), dim=-1)


def mixture_of_gaussians(x: torch.Tensor, mixtures, degree, sigma):
    shape = x.shape
    d = shape[-1]
    toreturn = torch.zeros(*shape[:-1])
    for i in range(mixtures):
        thisdeg = min(degree, d-i)
        mu = torch.zeros(degree)
        mu[0] = 1
        parts = x.index_select(dim=-1, index=torch.tensor([idx for idx in range(i, i+thisdeg)]))
        sqdist = torch.norm(parts - mu, dim=-1).pow(2)
        gauss = torch.exp(-1/(2*sigma**2) * sqdist) * i
        toreturn += gauss
    return -toreturn


def random_branin(x: torch.Tensor):
    d = x.shape[-1]
    torch.random.manual_seed(123456)
    i = torch.randint(0, d)
    j = torch.randint(0, d-1)
    if j >= i:  # easy way to choose 2 numbers w/o replacement
        j += 1
    return branin(x.index_select(dim=-1, index=torch.tensor([i,j])))


def branin(x: torch.Tensor):
    x1 = x.index_select(-1, torch.tensor(0))
    x2 = x.index_select(-1, torch.tensor(1))
    comp1 = (x2 - 5.1 / (4 * pi**2) * x1.pow(2) + 5/pi * x1 - 6).pow(2)
    comp2 = 10*(1-1/(8*pi))*torch.cos(x1) + 10
    return comp1 + comp2


def EI(xs, gp, best):
    """Expected improvement function (for minimization of a function)"""
    gp.eval()
    dist = gp(xs)
    mu = dist.mean
    cov = dist.covariance_matrix
    diagdist = torch.distributions.Normal(mu, cov.diag())
    cdf = diagdist.cdf(torch.tensor([best] * len(xs)))
    t1 = (best - mu) * cdf
    t2 = cov.diag() * torch.exp(diagdist.log_prob(best))
    return t1 + t2


class BayesOpt(object):
    def __init__(self, obj_fxn: Callable, bounds: Iterable, gp_model: gpytorch.models.GP, acq_fxn: Callable,
                 optimizer=differential_evolution, initial_point=None, gp_optim_freq=None, gp_optim_options=None):
        self.obj_fxn = obj_fxn
        self.bounds = bounds
        self.model = gp_model
        self.acq_fxn = acq_fxn
        self.optimizer = optimizer

        if initial_point is None:
            initial_point = torch.tensor(bounds, dtype=torch.float)
            initial_point = initial_point.mean(dim=1, keepdim=False)
            initial_point = initial_point.unsqueeze(0)

        self.obsX = initial_point
        # TODO: move this acquisition outside of the init fxn?
        self.obsY = obj_fxn(initial_point)
        self._best_x = self.obsX[0]
        self._best_y = self.obsY[0]
        self._n = 1

        self.gp_optim_freq = gp_optim_freq
        self.gp_optim_options = gp_optim_options

        self.update_model()

    def update_model(self):
        self.model.set_train_data(self.obsX, self.obsY, strict=False)
        if self.gp_optim_freq is not None and (self._n % self.gp_optim_freq == 0):
            self.model.train()
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)
            gp_models.train_to_convergence(self.model, self.obsX, self.obsY, objective=mll,
                                           **self.gp_optim_options)
        self.model.eval()
        return self.model

    def step(self):
        def optim_obj(x: np.array):
            x = torch.tensor([x], dtype=torch.float)
            return -self.acq_fxn(x, self.model, self._best_y).item()

        res = self.optimizer(optim_obj, self.bounds)  # TODO: look at more options?

        newX = torch.tensor([res.x], dtype=torch.float)
        newY = self.obj_fxn(newX)
        # newY = torch.tensor([res.fun], dtype=torch.float)
        if newY < self._best_y:
            self._best_y = newY
            self._best_x = newX
        self.obsX = torch.cat([self.obsX, newX], dim=0)
        self.obsY = torch.cat([self.obsY, newY], dim=0)
        self._n += 1

        self.update_model()
         # TODO: Make more verbose and store history
        return self

    def steps(self, num_steps):
        for _ in range(num_steps):
            self.step()
        return self
