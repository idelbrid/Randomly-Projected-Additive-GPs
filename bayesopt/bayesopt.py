import torch
import gpytorch
from typing import Callable, Iterable, Any
from .optimize import quasirandom_candidates, random_candidates, brute_candidates
import numpy as np
from botorch.models.model import Model
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch import fit_gpytorch_model
from botorch.optim.fit import fit_gpytorch_torch
from botorch.optim import joint_optimize
from .utils import get_lengthscales, get_mixins, format_for_str, get_outputscale

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
    def __init__(self, obj_fxn: Callable,
                 bounds: Iterable,
                 init_model_fxn: Callable[[torch.Tensor, torch.Tensor], Model],
                 acq_fxn: AcquisitionFunction,
                 initial_points=10,
                 init_method='latin_hc',
                 gp_optim_freq=None,
                 from_scratch_freq=1e5,
                 ):
        self.obj_fxn = obj_fxn
        self.bounds = torch.as_tensor(bounds, dtype=torch.float)

        self.init_model_fxn = init_model_fxn
        self.model = None
        self.acq_fxn = acq_fxn

        self._dimension = self.bounds.shape[0]
        self._internal_bounds = torch.tensor([[0, 1] for _ in range(self._dimension)], dtype=torch.float).t()
        self._initial_points = initial_points
        self._init_method = init_method

        self.obsX = torch.tensor([], dtype=torch.float)
        self.obsY = torch.tensor([], dtype=torch.float)
        self.true_y_history = torch.tensor([], dtype=torch.float)
        self._obsY_std = None
        self._n = 0

        self._initialized = False
        self.gp_optim_freq = gp_optim_freq
        self.from_scratch_freq = from_scratch_freq

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
        self.update_model(manually_set=True, force_update=True, from_scratch=True)

    @property
    def _best_internal_y(self):
        return self.obsY.max()

    @property
    def _best_internal_x(self):
        return self.obsX[self.obsY.argmax(), :]

    @property
    def best_true_y(self):
        return self.true_y_history.max()

    @property
    def best_true_y_path(self):
        return [self.true_y_history[:i+1].max().item() for i in range(self.true_y_history.shape[0])]

    def update_model(self, manually_set=True, force_update=False, from_scratch=False):
        # SingleTaskGP()
        # model = FixedNoiseGP(train_x, train_obj, train_yvar.expand_as(train_obj)).to(train_x)
        # mll = ExactMarginalLogLikelihood(model.likelihood, model)

        need_to_refit = force_update or (self.gp_optim_freq is not None and (self._n % self.gp_optim_freq == 0))
        if need_to_refit:
            self._obsY_std = self.true_y_history.std().detach()
            self.obsY = self.descale_y(self.true_y_history)

        if need_to_refit and from_scratch:
            # create entirely new GP model using user provided hook
            self.model = self.init_model_fxn(self.obsX, self.obsY)
        elif manually_set or need_to_refit:
            # manually set the training data of an existing model
            self.model.set_train_data(self.obsX, self.obsY, strict=False)
        else:
            # just get the fantasy model
            newX = self.obsX[-1:-2, :]
            newY = self.obsY[-1:-2]
            self.model = self.model.get_fantasy_model(newX, newY)

        if need_to_refit:
            self.model.train()
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)
            fit_gpytorch_model(mll, optimizer=fit_gpytorch_torch)
            # gp_models.train_to_convergence(self.model, self.obsX, self.obsY,
            #                                objective=mll, **self.gp_optim_options)
        self.model.eval()
        return self.model

    def _maximize_acq(self, num_restarts=10, raw_samples=500):
        acq_func = self.acq_fxn(self.model, self._best_internal_y)
        candidates = joint_optimize(
            acq_function=acq_func,
            bounds=self._internal_bounds,  # I think this should actually just be like [0,1]
            num_restarts=num_restarts,
            q=1,
            raw_samples=raw_samples,  # used for initialization heuristic
        )
        # observe new values
        new_x = candidates.detach()
        return new_x

    def step(self, **kwargs):
        if not self._initialized:
            raise ValueError("Optimizer not initialized! Initialize first")

        new_x = self._maximize_acq(**kwargs)
        Y_at_scale = self.obj_fxn(scale_to_bounds(new_x, self.bounds))
        # Store the true Y value before scaling down to current std
        self.true_y_history = torch.cat([self.true_y_history, Y_at_scale], dim=0)

        newY = self.descale_y(Y_at_scale)
        self.obsX = torch.cat([self.obsX, new_x], dim=0)
        self.obsY = torch.cat([self.obsY, newY], dim=0)
        self._n += len(newY)

        self.update_model(from_scratch=self._n % self.from_scratch_freq == 0)

        return self

    def steps(self, num_steps, print_=True, **optimizer_kwargs):
        if print_:
            print(self.best_true_y)
        for i in range(num_steps):
            self.step(**optimizer_kwargs)
            if print_:
                print(i, self.best_true_y, '\n\toutputscale={}, \n\tlengthscale(s)={}, \n\tmixins={}'.format(
                    format_for_str(get_outputscale(self.model.covar_module)),
                    format_for_str(get_lengthscales(self.model.covar_module)),
                    format_for_str(get_mixins(self.model.covar_module))
                ))
        return self


