import torch
import gpytorch
from math import pi
from typing import Callable, Iterable
from scipy.optimize import differential_evolution
import gp_models
from gp_models import AdditiveKernel, AdditiveExactGPModel, StrictlyAdditiveKernel, ProjectedAdditiveExactGPModel
import numpy as np
from torch.quasirandom import SobolEngine
import rp
import scipy


def easy_meshgrid(sizes, numpy=False, interior=True):
    spot_per_dim = []
    for i in range(len(sizes)):
        if interior:
            spots = torch.linspace(0, 1, sizes[i] + 2)[1:-1]
        else:
            spots = torch.linspace(0, 1, sizes[i])
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

    def _maximize_nonadditive_acq(self, **kwargs):
        if self.optimizer not in ['brute', 'random', 'quasirandom']:
            best_y = self._best_internal_y

            def optim_obj(x: np.array):
                x = torch.tensor([x], dtype=torch.float)
                return -self.acq_fxn(x, self.model, best_y).item()

            res = self.optimizer(optim_obj, [[0, 1] for _ in range(self._dimension)],
                                 **kwargs)  # TODO: look at more options?
            res.x = torch.tensor([res.x])  # wait what?
        else:
            acq = self.acq_fxn(self.candX(), self.model, self._best_internal_y)
            max_idx = acq.argmax()
            max_x = self.candX()[max_idx:max_idx + 1]
            max_acq = acq[max_idx]
            res = scipy.optimize.OptimizeResult(x=max_x, fun=max_acq)
        return res

    def _maximize_projected_additive_acq(self, **kwargs):
        opt_model, proj = self.model.get_corresponding_additive_model(return_proj=True)
        # Assuming the projection is linear
        # NB: torch.nn.Linear(n,m).weight is an m x n matrix. So it's the transpose of the matrix we're trying to use.
        with torch.no_grad():
            b = torch.clamp(proj.weight, min=0).sum(dim=1)
            a = torch.clamp(proj.weight, max=0).sum(dim=1)
            opt_dim = proj.weight.shape[1]
        bounds = [[a[i], b[i]] for i in range(len(a))]

        # TODO: instead of copying code actually write re-usable code.
        if self.optimizer not in ['brute', 'random', 'quasirandom']:
            group_results = []

            for i, group in enumerate(opt_model.get_groups()):
                group_bounds = [bounds[i] for i in group]
                def optim_obj(x: np.array):
                    x = torch.tensor([x], dtype=torch.float)
                    return -self.acq_fxn(x, opt_model, i).item() #  a surrogate actually
                group_res = self.optimizer(optim_obj, group_bounds, **kwargs)
                group_res.x = torch.tensor([group_res.x])
                group_results.append(group_res)
        else:
            group_results = []
            for i, group in enumerate(opt_model.get_groups()):
                group_bounds = [bounds[i] for i in group]
                cands = self.candX(dim=len(group))
                cands = scale_to_bounds(cands, group_bounds)
                group_acq = self.acq_fxn(cands, opt_model, i)
                max_idx = group_acq.argmax()
                max_x = cands[max_idx]
                max_acq = group_acq[max_idx]
                group_res = scipy.optimize.OptimizeResult(x=max_x, fun=max_acq)
                group_results.append(group_res)

        z_results = aggregate_results(group_results, opt_dim, opt_model.get_groups())
        with torch.no_grad():
            z = z_results.x
            x = z.matmul(torch.inverse(proj.weight.t()))
            # NB: torch.nn.Linear(n,m).weight is an m x n matrix. So it's the transpose of the matrix we're trying to use.
        z_results.z = z
        z_results.x = x
        return z_results

    def _maximize_additive_acq(self, **kwargs):

        if self.optimizer not in ['brute', 'random', 'quasirandom']:
            group_results = []
            for i, group in enumerate(self.model.get_groups()):
                def optim_obj(x: np.array):
                    x = torch.tensor([x], dtype=torch.float)
                    return -self.acq_fxn(x, self.model, i).item() #  a surrogate actually
                group_res = self.optimizer(optim_obj, [[0, 1] for _ in range(len(group))], **kwargs)
                group_res.x = torch.tensor([group_res.x])
                group_results.append(group_res)
            # TODO: recombine results
        else:
            group_results = []
            for i, group in enumerate(self.model.get_groups()):
                cands = self.candX(dim=len(group))
                group_acq = self.acq_fxn(cands, self.model, i)
                max_idx = group_acq.argmax()
                max_x = cands[max_idx]
                max_acq = group_acq[max_idx]
                group_res = scipy.optimize.OptimizeResult(x=max_x, fun=max_acq)
                group_results.append(group_res)

        return aggregate_results(group_results, self._dimension, self.model.get_groups())

    def _maximize_acq(self, **kwargs):
        if not self._acq_fxn_name.lower().startswith('add'):
            return self._maximize_nonadditive_acq(**kwargs)
        else:
            if isinstance(self.model, AdditiveExactGPModel):
                return self._maximize_additive_acq(**kwargs)
            else:
                return self._maximize_projected_additive_acq(**kwargs)

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
    from scipydirect import minimize as DIRectMinimize
    d = 10
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
    kernel = training_routines.create_rp_poly_kernel(d, 1, d, None, learn_proj=True, weighted=True,
                                                     kernel_type='RBF', space_proj=False)
    kernel.projection_module.weight.data = rp.gen_rp(d, d)
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
