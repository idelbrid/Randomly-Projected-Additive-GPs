import numpy as np
import torch
from gpytorch.models import GP
from torch.quasirandom import SobolEngine

#
# def maximize_enumerate(model: GP,
#                        acquisition: torch.nn.Module,
#                        method='random',
#                        n_samples=2000,
#                        **kwargs):
#     dim = model.traing_inputs.shape[-1]
#     if method == 'random':
#         samples = random_candidates(n_samples, dim)  # might be wrong with training inputs.
#     elif method == 'brute':
#         samples = brute_candidates(n_samples, dim)
#     elif method == 'quasirandom':
#         samples = quasirandom_candidates(n_samples, dim)
#     else:
#         raise ValueError("Invalid enumeration method for brute force global optimization of acquisition.")
#     model.eval()
#     posterior = model(samples)
#     acq_values = acquisition(posterior, **kwargs)
#     idx = acq_values.argmax()
#     return samples[idx], acq_values[idx]
#
#     #
#     # def optim_obj(x: np.array):
#     #     x = torch.tensor([x], dtype=torch.float)
#     #     return -self.acq_fxn(x, self.model, best_y).item()
#     #
#     # res = self.optimizer(optim_obj, [[0, 1] for _ in range(self._dimension)],
#     #                      **kwargs)  # TODO: look at more options?
#     # res.x = torch.tensor([res.x])  # wait what?
#
#
# def maximize_scipy():
#     options = options or {}
#     clamped_candidates = columnwise_clamp(
#         initial_conditions, lower_bounds, upper_bounds
#     ).requires_grad_(True)
#
#     shapeX = clamped_candidates.shape
#     x0 = _arrayify(clamped_candidates.view(-1))
#     bounds = make_scipy_bounds(
#         X=initial_conditions, lower_bounds=lower_bounds, upper_bounds=upper_bounds
#     )
#     constraints = make_scipy_linear_constraints(
#         shapeX=clamped_candidates.shape,
#         inequality_constraints=inequality_constraints,
#         equality_constraints=equality_constraints,
#     )
#
#     def f(x):
#         X = (
#             torch.from_numpy(x)
#             .to(initial_conditions)
#             .view(shapeX)
#             .contiguous()
#             .requires_grad_(True)
#         )
#         X_fix = fix_features(X=X, fixed_features=fixed_features)
#         loss = -acquisition_function(X_fix).sum()
#         loss.backward()
#         fval = loss.item()
#         gradf = _arrayify(X.grad.view(-1))
#         return fval, gradf
#
# def maximize_grad():
#     pass
#
# def maximize_add_scipy():
#     pass
#
# def maximize_add_enumerate():
#     pass
#
# def maximize_add_grad():
#     pass
#
# def _maximize_nonadditive_acq(self, **kwargs):
#     if self.optimizer not in ['brute', 'random', 'quasirandom']:
#         pass
#     else:
#         acq = self.acq_fxn(self.candX(), self.model, self._best_internal_y)
#         max_idx = acq.argmax()
#         max_x = self.candX()[max_idx:max_idx + 1]
#         max_acq = acq[max_idx]
#         res = scipy.optimize.OptimizeResult(x=max_x, fun=max_acq)
#     return res
#
#
# def _maximize_projected_additive_acq(self, **kwargs):
#     opt_model, proj = self.model.get_corresponding_additive_model(return_proj=True)
#     # Assuming the projection is linear
#     # NB: torch.nn.Linear(n,m).weight is an m x n matrix. So it's the transpose of the matrix we're trying to use.
#     with torch.no_grad():
#         b = torch.clamp(proj.weight, min=0).sum(dim=1)
#         a = torch.clamp(proj.weight, max=0).sum(dim=1)
#         opt_dim = proj.weight.shape[1]
#     bounds = [[a[i], b[i]] for i in range(len(a))]
#
#     # TODO: instead of copying code actually write re-usable code.
#     if self.optimizer not in ['brute', 'random', 'quasirandom']:
#         group_results = []
#
#         for i, group in enumerate(opt_model.get_groups()):
#             group_bounds = [bounds[i] for i in group]
#
#             def optim_obj(x: np.array):
#                 x = torch.tensor([x], dtype=torch.float)
#                 return -self.acq_fxn(x, opt_model, i).item()  # a surrogate actually
#
#             group_res = self.optimizer(optim_obj, group_bounds, **kwargs)
#             group_res.x = torch.tensor([group_res.x])
#             group_results.append(group_res)
#     else:
#         group_results = []
#         for i, group in enumerate(opt_model.get_groups()):
#             group_bounds = [bounds[i] for i in group]
#             cands = self.candX(dim=len(group))
#             cands = scale_to_bounds(cands, group_bounds)
#             group_acq = self.acq_fxn(cands, opt_model, i)
#             max_idx = group_acq.argmax()
#             max_x = cands[max_idx]
#             max_acq = group_acq[max_idx]
#             group_res = scipy.optimize.OptimizeResult(x=max_x, fun=max_acq)
#             group_results.append(group_res)
#
#     z_results = aggregate_results(group_results, opt_dim, opt_model.get_groups())
#     with torch.no_grad():
#         z = z_results.x
#         x = z.matmul(torch.inverse(proj.weight.t()))
#         # NB: torch.nn.Linear(n,m).weight is an m x n matrix. So it's the transpose of the matrix we're trying to use.
#     z_results.z = z
#     z_results.x = x
#     return z_results
#
#
# def _maximize_additive_acq(self, **kwargs):
#     if self.optimizer not in ['brute', 'random', 'quasirandom']:
#         group_results = []
#         for i, group in enumerate(self.model.get_groups()):
#             def optim_obj(x: np.array):
#                 x = torch.tensor([x], dtype=torch.float)
#                 return -self.acq_fxn(x, self.model, i).item()  # a surrogate actually
#
#             group_res = self.optimizer(optim_obj, [[0, 1] for _ in range(len(group))], **kwargs)
#             group_res.x = torch.tensor([group_res.x])
#             group_results.append(group_res)
#         # TODO: recombine results
#     else:
#         group_results = []
#         for i, group in enumerate(self.model.get_groups()):
#             cands = self.candX(dim=len(group))
#             group_acq = self.acq_fxn(cands, self.model, i)
#             max_idx = group_acq.argmax()
#             max_x = cands[max_idx]
#             max_acq = group_acq[max_idx]
#             group_res = scipy.optimize.OptimizeResult(x=max_x, fun=max_acq)
#             group_results.append(group_res)
#
#     return aggregate_results(group_results, self._dimension, self.model.get_groups())


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