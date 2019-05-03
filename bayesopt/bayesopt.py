import torch
import gpytorch
from typing import Callable, Iterable
from scipy.optimize import differential_evolution
import gp_models
from .acquisition import EI, ThompsonSampling, UCB, surrogate_AddUCB
from .utils import stybtang
from gp_models import AdditiveExactGPModel, ProjectedAdditiveExactGPModel
from .optimize import maximize_enumerate, maximize_grad, maximize_scipy
import numpy as np
from torch.quasirandom import SobolEngine


# TODO: implement Add-EI
# TODO: implement similar for random projections.
# TODO: move to construct entirely new GP at that time.
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


def aggregate_results(group_results, dimension, groups):
    # assume additive decomposition of the acquisition function
    actual_x = torch.empty(1, dimension, dtype=torch.float)
    actual_acq = 0
    for i, group in enumerate(groups):
        actual_acq += group_results[i].fun
        for j, index in enumerate(group):
            actual_x[:, index] = group_results[i].x[j]
    return scipy.optimize.OptimizeResult(x=actual_x, fun=actual_acq)


# TODO: add gradients
# TODO: add marginalization
class BayesOpt(object):
    def __init__(self, obj_fxn: Callable, bounds: Iterable, gp_model: gpytorch.models.GP, acq_fxn: str,
                 optimizer='quasirandom', initial_points=10, init_method='latin_hc', gp_optim_freq=None,
                 gp_optim_options=None):
        self.obj_fxn = obj_fxn
        self.bounds = bounds
        self.model = gp_model
        self._acq_fxn_name = acq_fxn
        if self._acq_fxn_name.lower().startswith('ei'):
            self.acq_fxn = EI
        elif self._acq_fxn_name.lower().startswith('thompson'):
            self.acq_fxn = ThompsonSampling
        elif self._acq_fxn_name.lower().startswith('ucb'):
            self.acq_fxn = UCB
        elif self._acq_fxn_name.lower().startswith('add_ucb'):
            self.acq_fxn = surrogate_AddUCB
        self.optimizer = optimizer
        self._dimension = len(bounds)
        self._initial_points = initial_points
        self._init_method = init_method

        self.obsX = torch.tensor([], dtype=torch.float)
        self.obsY = torch.tensor([], dtype=torch.float)
        self.true_y_history = []
        self._obsY_std = None
        self._n = 0
        self._candX = None
        self._num_candidates = 1500

        self._initialized = False
        self.gp_optim_freq = gp_optim_freq
        self.gp_optim_options = gp_optim_options

    def descale_y(self, y):
        return y / self._obsY_std

    def initialize(self):
        if self._init_method == 'random':
            self.obsX = random_candidates(self._initial_points, self._dimension)
        elif self._init_method == 'latin_hc':
            self.obsX = latin_hc_sampling(self._dimension, self._initial_points)

        self.obsY = self.obj_fxn(scale_to_bounds(self.obsX, self.bounds))
        self.true_y_history = self.obsY.clone().detach()
        self._obsY_std = self.obsY.std().detach()

        self.obsY = self.descale_y(self.obsY)
        self._initialized = True
        self.update_model(manually_set=True, force_update=True)

    @property
    def _best_internal_y(self):
        return self.obsY.min()

    @property
    def _best_internal_x(self):
        return self.obsX[self.obsY.argmin(), :]

    @property
    def best_true_y(self):
        return self.true_y_history.min()

    @property
    def best_true_y_path(self):
        return [self.true_y_history[:i+1].min().item() for i in range(len(self.true_y_history))]

    def candX(self, dim=None):
        if dim is None:
            dimension = self._dimension
        else:
            dimension = dim
        if self._candX is None or self.optimizer == 'random' or (dim is not None):
            if self.optimizer == 'quasirandom':
                candX = quasirandom_candidates(self._num_candidates, dimension)
            elif self.optimizer == 'random':
                candX = random_candidates(self._num_candidates, dimension)
            else:
                candX = brute_candidates(self._num_candidates, dimension)
            self._candX = candX

        return self._candX

    def update_model(self, manually_set=True, force_update=False):
        need_to_refit = force_update or (self.gp_optim_freq is not None and (self._n % self.gp_optim_freq == 0))
        if need_to_refit:
            self._obsY_std = self.true_y_history.std().detach()
            self.obsY = self.descale_y(self.true_y_history)

        if manually_set or need_to_refit:
            self.model.set_train_data(self.obsX, self.obsY, strict=False)
        else:
            newX = self.obsX[-1:-2, :]
            newY = self.obsY[-1:-2]
            self.model = self.model.get_fantasy_model(newX, newY)

        if need_to_refit:
            self.model.train()
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)
            gp_models.train_to_convergence(self.model, self.obsX, self.obsY,
                                           objective=mll, **self.gp_optim_options)
        self.model.eval()
        return self.model

    def _maximize_acq(self, **kwargs):
        # TODO: unbreak
        pass

    def step(self, **optimizer_kwargs):
        if not self._initialized:
            raise ValueError("Optimizer not initialized! Initialize first")

        res = self._maximize_acq(**optimizer_kwargs)

        newX = torch.tensor(res.x, dtype=torch.float)
        Y_at_scale = self.obj_fxn(scale_to_bounds(newX, self.bounds))
        # Store the true Y value before scaling down to current std
        self.true_y_history = torch.cat([self.true_y_history, Y_at_scale], dim=0)

        # Scale by standard deviation and add to internal data.
        newY = self.descale_y(Y_at_scale)
        self.obsX = torch.cat([self.obsX, newX], dim=0)
        self.obsY = torch.cat([self.obsY, newY], dim=0)
        self._n += 1
        self.update_model()
        return self

    def steps(self, num_steps, print_=True, **optimizer_kwargs):
        if print_:
            print(self.best_true_y)
        for i in range(num_steps):
            self.step(**optimizer_kwargs)
            if print_:
                print(i, self.best_true_y, 'noise {:2.3f}, outputscale={:3.1f}'.format(
                    self.model.likelihood.noise.item(),
                    self.model.covar_module.outputscale.item()))
                # print(self._internal_best_y, self._obsY_std, self._internal_best_y*self._obsY_std, self.true_best_y_path[-1])
        return self


if __name__ == '__main__':
    import training_routines
    import scipy

    d = 20
    bounds = [(-4, 4) for _ in range(d)]
    def objective_function(x):
        print('queries x=', x)
        res = stybtang(x)
        print('Obtained', res)
        return res

    trainX = torch.tensor([], dtype=torch.float)
    trainY = torch.tensor([], dtype=torch.float)
    lik = gpytorch.likelihoods.GaussianLikelihood()
    lik.initialize(noise=0.01)
    lik.raw_noise.requires_grad = False

    # kernel = training_routines.create_general_rp_poly_kernel(10, [1 for _ in range(d)], learn_proj=False,
    #                                                          kernel_type='Matern', weighted=True)
    kernel = training_routines.create_rp_poly_kernel(d, 1, d, None, learn_proj=False, weighted=True,
                                                     kernel_type='RBF', space_proj=True)
    # kernel.projection_module.weight.data = rp.gen_rp(d, d, 'very-sparse')
    # kernel = training_routines.create_full_kernel(d, ard=True, kernel_type='Matern')
    # kernel = training_routines.create_additive_kernel(d, kernel_type='Matern')
    kernel = gpytorch.kernels.ScaleKernel(kernel)
    # model = AdditiveExactGPModel(trainX, trainY, lik, kernel)
    model = ProjectedAdditiveExactGPModel(trainX, trainY, lik, kernel)
    inner_optimizer = 'quasirandom'
    gp_optim_options = dict(max_iter=20, optimizer=torch.optim.Adam, verbose=False, lr=0.1, check_conv=False)

    optimizer = BayesOpt(objective_function, bounds, model, 'add_ucb', inner_optimizer,
                            gp_optim_freq=5, gp_optim_options=gp_optim_options, initial_points=10)

    with gpytorch.settings.lazily_evaluate_kernels(True):
        optimizer.initialize()
        optimizer.steps(200, maxf=200)
