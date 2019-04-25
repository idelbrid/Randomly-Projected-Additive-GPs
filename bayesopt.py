import torch
import gpytorch
from math import pi
from typing import Callable, Iterable
from scipy.optimize import differential_evolution
import gp_models
import numpy as np
from torch.quasirandom import SobolEngine
import scipy


def easy_meshgrid(sizes, numpy=False):
    spot_per_dim = []
    for i in range(len(sizes)):
        spots = torch.linspace(0, 1, sizes[i] + 2)[1:-1]
        spot_per_dim.append(spots)
    tensors = torch.meshgrid(spot_per_dim)
    stacked = torch.stack(tensors)
    res = stacked.reshape(len(sizes), -1).t()
    if numpy:
        res = res.numpy()
    return res


def stybtang(x: torch.Tensor):
    return 1/2 * torch.sum(x.pow(4) - 16*x.pow(2) + 5*x, dim=-1)


def np_stybtang(x: np.array):
    return 1/2 * np.sum(np.power(x, 4) - 16*np.power(x, 2) + 5 * x, axis=-1)


def michalewicz(x: torch.Tensor, m: int):
    d = x.shape[-1]
    scaled_x = x * torch.arange(1, d+1, dtype=torch.float).unsqueeze(0)
    return -torch.sum(torch.sin(x) * torch.sin(scaled_x / pi).pow(2*m), dim=-1)


def np_michalewicz(x: np.array, m: int):
    d = x.shape[-1]
    scaled_x = x * np.arange(1, d+1, dtype=np.float)
    return -np.sum(np.sin(x) * np.power(np.sin(scaled_x / pi), 2*m), axis=-1)


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


def EI(xs, gp, best, use_love=False):
    """Expected improvement function (for minimization of a function)"""
    gp.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var(use_love):
        dist = gp(xs)
        mu = dist.mean
        cov = dist.covariance_matrix
        diagdist = torch.distributions.Normal(mu, cov.diag())
        cdf = diagdist.cdf(torch.tensor([best] * len(xs)))
        t1 = (best - mu) * cdf
        t2 = cov.diag() * torch.exp(diagdist.log_prob(best))
        return t1 + t2


def ThompsonSampling(xs, gp, best, use_love=False):
    gp.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_samples(use_love):
        dist = gp(xs)
        samples = dist.sample()
    return -samples


# TODO: implement UCB, Add-UCB, and Add-EI
# TODO: implement latin hypercube sampling
# TODO: move to initialize with ~10 samples from latin hypercube
# TODO: move to updating only every so often, and to construct entirely new GP at that time.
# TODO: move to using DIRect
# TODO: include native scaling of the GP
# TODO: implement neat .save function to make this shit easier to document.


def scale_to_bounds(x, bounds):
    left = torch.tensor([b[0] for b in bounds], dtype=torch.float).unsqueeze(dim=0)
    right = torch.tensor([b[1] for b in bounds], dtype=torch.float).unsqueeze(dim=0)
    return x * (right - left) + left


def unscale_from_bounds(x, bounds):
    left = torch.tensor([b[0] for b in bounds], dtype=torch.float).unsqueeze(dim=0)
    right = torch.tensor([b[1] for b in bounds], dtype=torch.float).unsqueeze(dim=0)
    return (x - left) / (right - left)


def quasirandom_candidates(n, dim):
    engine = SobolEngine(dim, scramble=True)
    candX = engine.draw(n)
    return candX


def random_candidates(n, dim):
    candX = torch.rand(n, dim)
    return candX


def brute_candidates(n, dim):
    spot_per_dim = []
    num_per_dim = int(np.floor(np.power(n, 1 / dim)))
    for i in range(dim):
        spots = torch.linspace(0, 1, num_per_dim + 2)[1:-1]
        spot_per_dim.append(spots)
    tensors = torch.meshgrid(spot_per_dim)
    stacked = torch.stack(tensors)
    candX = stacked.reshape(-1, dim)
    return candX


# NOTE: adapted from dragonfly_opt (https://github.com/dragonfly/dragonfly/blob/master/dragonfly/utils/oper_utils.py)
# Utilities for sampling from combined domains ===================================
def _latin_hc_indices(dim, num_samples):
  """ Obtains indices for Latin Hyper-cube sampling. """
  index_set = [list(range(num_samples))] * dim
  lhs_indices = []
  for i in range(num_samples):
    curr_idx_idx = np.random.randint(num_samples-i, size=dim)
    curr_idx = [index_set[j][curr_idx_idx[j]] for j in range(dim)]
    index_set = [index_set[j][:curr_idx_idx[j]] + index_set[j][curr_idx_idx[j]+1:]
                 for j in range(dim)]
    lhs_indices.append(curr_idx)
  return lhs_indices

def latin_hc_sampling(dim, num_samples):
  """ Latin Hyper-cube sampling in the unit hyper-cube. """
  if num_samples == 0:
    return torch.zeros(0, dim)
  elif num_samples == 1:
    return 0.5 * torch.ones(1, dim)
  lhs_lower_boundaries = (torch.linspace(0, 1, num_samples+1)[:num_samples]).reshape(1, -1)
  width = lhs_lower_boundaries[0][1] - lhs_lower_boundaries[0][0]
  lhs_lower_boundaries = np.repeat(lhs_lower_boundaries, dim, axis=0).t()
  lhs_indices = _latin_hc_indices(dim, num_samples)
  lhs_sample_boundaries = []
  for i in range(num_samples):
    curr_idx = lhs_indices[i]
    curr_sample_boundaries = [lhs_lower_boundaries[curr_idx[j]][j] for j in range(dim)]
    lhs_sample_boundaries.append(curr_sample_boundaries)
  lhs_sample_boundaries = torch.tensor(lhs_sample_boundaries, dtype=torch.float)
  uni_random_width = width * torch.rand(num_samples, dim)
  lhs_samples = lhs_sample_boundaries + uni_random_width
  return lhs_samples


# TODO: add rescaling
# TODO: add gradients
# TODO: add marginalization
class BayesOpt(object):
    def __init__(self, obj_fxn: Callable, bounds: Iterable, gp_model: gpytorch.models.GP, acq_fxn: Callable,
                 optimizer='quasirandom', initial_points=10, init_method='latin_hc', gp_optim_freq=None,
                 gp_optim_options=None):
        self.obj_fxn = obj_fxn
        self.bounds = bounds
        self.model = gp_model
        self.acq_fxn = acq_fxn
        self.optimizer = optimizer
        self._dimension = len(bounds)
        self._initial_points = initial_points
        self._init_method = init_method

        self.obsX = torch.tensor([], dtype=torch.float)
        self.obsY = torch.tensor([], dtype=torch.float)
        self._best_x = None
        self._best_y = None
        self._best_y_path = []
        self._obsY_std = None
        self._n = 0
        self._candX = None
        self._num_candidates = 1500

        self._initialized = False
        self.gp_optim_freq = gp_optim_freq
        self.gp_optim_options = gp_optim_options

    def initialize(self):
        if self._init_method == 'random':
            self.obsX = random_candidates(self._initial_points, self._dimension)
        elif self._init_method == 'latin_hc':
            self.obsX = latin_hc_sampling(self._dimension, self._initial_points)

        self.obsY = self.obj_fxn(scale_to_bounds(self.obsX, self.bounds))
        best_idx = self.obsY.argmin()
        self._best_x = self.obsX[best_idx, :]
        self._best_y = self.obsY[best_idx]
        self._best_y_path = [self._best_y.item() for _ in range(self._initial_points)]
        self._initialized = True
        self.update_model(self.obsX, self.obsY, manually_set=True, force_update=True)

    def update_model(self, newX, newY, manually_set=True, force_update=False):
        if manually_set:
            self.model.set_train_data(self.obsX, self.obsY, strict=False)
        else:
            self.model = self.model.get_fantasy_model(newX, newY)

        if force_update or (self.gp_optim_freq is not None and (self._n % self.gp_optim_freq == 0)):
            self.model.train()
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)
            gp_models.train_to_convergence(self.model, self.obsX, self.obsY,
                                           objective=mll, **self.gp_optim_options)
        self.model.eval()
        return self.model

    @property
    def candX(self):
        if self._candX is None or self.optimizer == 'random':
            if self.optimizer == 'quasirandom':
                candX = quasirandom_candidates(self._num_candidates, self._dimension)
            elif self.optimizer == 'random':
                candX = random_candidates(self._num_candidates, self._dimension)
            else:
                candX = brute_candidates(self._num_candidates, self._dimension)
            self._candX = candX

        return self._candX

    def _scale_to_bounds(self, x): # TODO: remove
        return scale_to_bounds(x, self.bounds)

    def _maximize_acq(self, **kwargs):
        if self.optimizer not in ['brute', 'random', 'quasirandom']:
            def optim_obj(x: np.array):
                x = torch.tensor([x], dtype=torch.float)
                return -self.acq_fxn(x, self.model, self._best_y).item()
            res = self.optimizer(optim_obj, self.bounds, **kwargs)  # TODO: look at more options?
            res.x = torch.tensor([res.x])
        else:
            acq = self.acq_fxn(self.candX, self.model, self._best_y)
            maxidx = acq.argmax()
            maxx = self.candX[maxidx:maxidx+1]
            maxy = self.acq_fxn(maxx, self.model, self._best_y)
            res = scipy.optimize.OptimizeResult(x=maxx.numpy(), fun=maxy.numpy())
        return res

    def step(self, **optimizer_kwargs):
        if not self._initialized:
            raise ValueError("Optimizer not initialized! Initialize first")

        res = self._maximize_acq(**optimizer_kwargs)

        newX = torch.tensor(res.x, dtype=torch.float)
        newY = self.obj_fxn(scale_to_bounds(newX, self.bounds))

        if newY < self._best_y:
            self._best_y = newY
            self._best_x = newX
        self.obsX = torch.cat([self.obsX, newX], dim=0)
        self.obsY = torch.cat([self.obsY, newY], dim=0)
        self._n += 1
        self._best_y_path.append(self._best_y.item())

        self.update_model(newX, newY)
        return self

    def steps(self, num_steps, print_=True, **optimizer_kwargs):
        if print_:
            print(self._best_y)
        for i in range(num_steps):
            self.step(**optimizer_kwargs)
            if print_:
                print(i, self._best_y.item(), 'noise {:2.3f}, outputscale={:3.1f}'.format(
                    self.model.likelihood.noise.item(),
                    self.model.covar_module.outputscale.item())
        return self


# class AdditiveBayesOpt(BayesOpt):
#     def __init__(self, obj_fxn, bounds, gp_model, acq_fxn, additive_components, optimizer='quasirandom',
#                  initial_point=None, gp_optim_freq=None, gp_optim_options=None):
#         super(AdditiveBayesOpt, self).__init__(obj_fxn, bounds, gp_model, acq_fxn, optimizer=optimizer,
#                                                initial_point=initial_point, gp_optim_freq=gp_optim_freq,
#                                                gp_optim_options=gp_optim_options)
#         for i in range(self._dimension):
#             ct = 0
#             for comp in self.additive_components:
#                 for j in comp:
#                     if i == j:
#                         ct += 1
#             if ct > 1:
#                 raise ValueError("Multiple inclusions of parameter {}".format(i))
#             elif ct < 1:
#                 raise ValueError("Exclusion of parameter {}".format(i))
#
#         self.additive_components = additive_components
#         component_bounds = []
#         for comp in self.additive_components:
#             bds = []
#             for j in comp:
#                 bds.append(self.bounds[j])
#             component_bounds.append(bds)
#         self._component_bounds = component_bounds
#
#     def candX(self):
#         if self._candX is None or self.optimizer == 'random':
#             candidate_sets = []
#             n_cand_per = self._num_candidates // len(self.additive_components)
#             for i, comp in self.additive_components:
#                 if self.optimizer == 'quasirandom':
#                     candX = quasirandom_candidates(n_cand_per, self._component_bounds[i])
#                 elif self.optimizer == 'random':
#                     candX = random_candidates(n_cand_per, self._component_bounds[i])
#                 else:
#                     candX = brute_candidates(n_cand_per, self._component_bounds[i])
#                 candidate_sets.append(candX)
#
#             self._candX = candidate_sets
#         return self._candX
#
#     def _maximize_acq(self, **kwargs):
#         if self.optimizer not in ['brute', 'random', 'quasirandom']:
#             raise NotImplementedError("General numerical optimizers not supported yet for additive decomposition")
#             def optim_obj(x: np.array):
#                 self._evaluations += 1
#                 x = torch.tensor([x], dtype=torch.float)
#                 return -self.acq_fxn(x, self.model, self._best_y).item()
#
#             res = self.optimizer(optim_obj, self.bounds, **kwargs)  # TODO: look at more options?
#         else:
#             results = []
#             candidate_sets = self.candX
#             for i, comp in enumerate(self.additive_components):
#                 component_cands = torch.zeros(len(candidate_sets[i]), self._dimension)
#                 component_cands[:, comp] = candidate_sets[i]
#                 acq = self.acq_fxn(self.candX, self.model, self._best_y)
#                 maxidx = acq.argmax()
#                 maxx = self.candX[maxidx]
#                 maxy = self.obj_fxn(maxx)
#                 self._candX[maxidx, :] = scale_to_bounds(torch.rand(1, self._dimension), self.bounds)
#                 res = scipy.optimize.OptimizeResult(x=maxx.numpy(), fun=maxy.numpy)
#         return res


if __name__ == '__main__':
    import training_routines
    import scipy
    d = 10
    bounds = [(-4, 4) for _ in range(d)]
    objective_function = stybtang
    # kernel = training_routines.create_general_rp_poly_kernel(10, [1, 1, 1, 2, 2, 3], learn_proj=False,
    #                                                          kernel_type='Matern')
    kernel = training_routines.create_full_kernel(d, ard=True, kernel_type='Matern')
    kernel = gpytorch.kernels.ScaleKernel(kernel)

    trainX = torch.tensor([], dtype=torch.float)
    trainY = torch.tensor([], dtype=torch.float)
    lik = gpytorch.likelihoods.GaussianLikelihood()
    lik.initialize(noise=0.01)
    lik.raw_noise.requires_grad = False
    model = gp_models.ExactGPModel(trainX, trainY, lik, kernel)
    inner_optimizer = 'quasirandom'
    gp_optim_options = dict(max_iter=15, optimizer=torch.optim.Adam, verbose=False, lr=0.1, check_conv=False)

    optimizer = BayesOpt(objective_function, bounds, model, EI, inner_optimizer,
                            gp_optim_freq=1, gp_optim_options=gp_optim_options)

    with gpytorch.settings.lazily_evaluate_kernels(True):
        optimizer.steps(20)