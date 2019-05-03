
def maximize_enumerate():
    best_y = self._best_internal_y

    def optim_obj(x: np.array):
        x = torch.tensor([x], dtype=torch.float)
        return -self.acq_fxn(x, self.model, best_y).item()

    res = self.optimizer(optim_obj, [[0, 1] for _ in range(self._dimension)],
                         **kwargs)  # TODO: look at more options?
    res.x = torch.tensor([res.x])  # wait what?

def maximize_scipy():
    pass

def maximize_grad():
    pass

def maximize_add_scipy():
    pass

def maximize_add_enumerate():
    pass

def maximize_add_grad():
    pass

def _maximize_nonadditive_acq(self, **kwargs):
    if self.optimizer not in ['brute', 'random', 'quasirandom']:

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
                return -self.acq_fxn(x, opt_model, i).item()  # a surrogate actually

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
                return -self.acq_fxn(x, self.model, i).item()  # a surrogate actually

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