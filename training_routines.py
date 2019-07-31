import math
import rp
from gp_models import ExactGPModel, ProjectionKernel, \
    PolynomialProjectionKernel, DNN,\
    GeneralizedPolynomialProjectionKernel, GeneralizedProjectionKernel, StrictlyAdditiveKernel, AdditiveKernel, DuvenaudAdditiveKernel
from gp_models.kernels import ManualRescaleProjectionKernel, InverseMQKernel, MemoryEfficientGamKernel
import gpytorch
from gpytorch.kernels import ScaleKernel, RBFKernel, GridInterpolationKernel, MaternKernel, InducingPointKernel, MultiDeviceKernel
from gpytorch.mlls import VariationalELBO, VariationalMarginalLogLikelihood
from gp_models import SVGPRegressionModel
import torch
import warnings
import copy
import os
from config import data_base_path, model_base_path
import numpy as np
from itertools import combinations
from fitting.optimizing import train_to_convergence, mean_squared_error, learn_projections

def _map_to_optim(optimizer):
    """Helper to map optimizer string names to torch objects"""
    if optimizer == 'adam':
        optimizer_ = torch.optim.Adam
    elif optimizer == 'sgd':
        optimizer_ = torch.optim.SGD
    elif optimizer == 'lbfgs':
        optimizer_ = torch.optim.LBFGS
    else:
        raise ValueError("Unknown optimizer")
    return optimizer_


def _save_state_dict(model):
    """Helper to save the state dict of a torch model to a unique filename"""
    d = model.state_dict()
    s = str(d)
    h = hash(s)
    fname = 'model_state_dict_{}.pkl'.format(h)
    torch.save(d, os.path.join(model_base_path, 'models', fname))
    return fname


def _sample_from_range(num_samples, range_):
    return torch.rand(num_samples) * (range_[1] - range_[0]) + range_[0]


def create_deep_rp_poly_kernel(d, degrees, projection_architecture, projection_kwargs, learn_proj=False,
                               weighted=False, kernel_type='RBF', init_mixin_range=(1.0, 1.0),
                               init_lengthscale_range=(1.0, 1.0), ski=False, ski_options=None,
                               X=None):
    outputs = sum(degrees)
    if projection_architecture == 'dnn':
        module = DNN(d, outputs, **projection_kwargs)
    else:
        raise NotImplementedError("No architecture besides DNN is implemented ATM")

    if kernel_type == 'RBF':
        kernel = gpytorch.kernels.RBFKernel
        kwargs = dict()
    elif kernel_type == 'Matern':
        kernel = gpytorch.kernels.MaternKernel
        kwargs = dict(nu=1.5)
    else:
        raise ValueError("Unknown kernel type")

# <<<<<<< HEAD
    kernel = GeneralizedProjectionKernel(degrees, d, kernel, module,
                                                 learn_proj=learn_proj,
                                                 weighted=weighted, ski=ski, ski_options=ski_options, X=X,**kwargs)
# =======
#     kernel = GeneralizedPolynomialProjectionKernel(J, k, d, kernel, module,
#                                                    learn_proj=learn_proj,
#                                                    weighted=weighted, ski=ski, ski_options=ski_options, X=X,
# >>>>>>> 50d9d6c6c2338a0f712858232ea93dd4f3107921
#                                                    **kwargs)
    kernel.initialize(init_mixin_range, init_lengthscale_range)
    return kernel


def create_rp_poly_kernel(d, k, J, activation=None,
                          learn_proj=False, weighted=False, kernel_type='RBF',
                          space_proj=False, init_mixin_range=(1.0, 1.0), init_lengthscale_range=(1.0, 1.0),
                          ski=False, ski_options=None, X=None, proj_dist='gaussian',
                          ):
    projs = [rp.gen_rp(d, k, dist=proj_dist) for _ in range(J)]
    bs = [torch.zeros(k) for _ in range(J)]

    if space_proj:
        # TODO: If k>1, could implement equal spacing for each set of projs
        newW, _ = rp.space_equally(torch.cat(projs,dim=1).t(), lr=0.1, niter=5000)
        # newW = rp.compute_spherical_t_design(d-1, t=4, N=J)
        newW.requires_grad = False
        projs = [newW[i:i+1, :].t() for i in range (J)]

    if kernel_type == 'RBF':
        kernel = gpytorch.kernels.RBFKernel
        kwargs = dict()
    elif kernel_type == 'Matern':
        kernel = gpytorch.kernels.MaternKernel
        kwargs = dict(nu=1.5)
    elif kernel_type == 'Cosine':
        kernel = gpytorch.kernels.CosineKernel
    else:
        raise ValueError("Unknown kernel type")

    kernel = PolynomialProjectionKernel(J, k, d, kernel, projs, bs, activation=activation, learn_proj=learn_proj,
                                        weighted=weighted, ski=ski, ski_options=ski_options, X=X, **kwargs)
    kernel.initialize(init_mixin_range, init_lengthscale_range)
    return kernel

def create_additive_rp_kernel(d, J, learn_proj=False, kernel_type='RBF', space_proj=False, prescale=False, ard=True,
                              init_lengthscale_range=(1., 1.), ski=False, ski_options=None, proj_dist='gaussian',
                              batch_kernel=True, mem_efficient=False, k=1):
    if k > 1 and (mem_efficient or batch_kernel or space_proj):
        raise ValueError("Can't have k > 1 with memory efficient GAM kernel or a batch kernel or spaced projections.")

    projs = [rp.gen_rp(d, k, dist=proj_dist) for _ in range(J)]
    # bs = [torch.zeros(1) for _ in range(J)]
    if space_proj:
        newW, _ = rp.space_equally(torch.cat(projs,dim=1).t(), lr=0.1, niter=5000)
        # newW = rp.compute_spherical_t_design(d-1, N=J)
        newW.requires_grad = False
        projs = [newW[i:i+k, :].t() for i in range(0, J*k, k)]
    proj_module = torch.nn.Linear(d, J*k, bias=False)
    proj_module.weight.data = torch.cat(projs, dim=1).t()
    # proj_module.bias.data = torch.cat(bs, dim=0)

    def make_kernel(active_dim=None):
        if kernel_type == 'RBF':
            kernel = gpytorch.kernels.RBFKernel(active_dims=active_dim)
        elif kernel_type == 'Matern':
            kernel = gpytorch.kernels.MaternKernel(nu=1.5, active_dims=active_dim)
        elif kernel_type == 'Cosine':
            kernel = gpytorch.kernels.CosineKernel(active_dims=active_dim)
        else:
            raise ValueError("Unknown kernel type")
        if hasattr(kernel, 'period_length'):
            kernel.initialize(period_length=torch.tensor([1.]))
        else:
            kernel.initialize(lengthscale=torch.tensor([1.]))
        kernel = ScaleKernel(kernel)
        kernel.initialize(outputscale=torch.tensor([1/J]))
        if ski:
            kernel = gpytorch.kernels.GridInterpolationKernel(kernel, **ski_options)
        return kernel

    if mem_efficient:
        if ski:
            raise ValueError("Not implemented yet")
        if batch_kernel:
            raise ValueError("Impossible to have batch kernel and memory efficient GAM")
        if kernel_type != 'RBF':
            raise ValueError("Memory efficient GAM with alternative sub-kernels not implemented yet.")
        add_kernel = MemoryEfficientGamKernel()
    elif batch_kernel:
        kernel = make_kernel(None)
        add_kernel = gpytorch.kernels.AdditiveStructureKernel(kernel, J)
    else:
        kernels = [make_kernel(list(range(i, i+k))) for i in range(0, J*k, k)]
        add_kernel = gpytorch.kernels.AdditiveKernel(*kernels)
    if ard:
        if prescale:
            ard_num_dims = d
            # print('prescaling')
        else:
            ard_num_dims = J*k
        initial_ls = _sample_from_range(ard_num_dims, init_lengthscale_range)
    else:
        ard_num_dims = None
        initial_ls = _sample_from_range(1, init_lengthscale_range)

    proj_kernel = ManualRescaleProjectionKernel(proj_module, add_kernel, prescale=prescale, ard_num_dims=ard_num_dims,
                                                learn_proj=learn_proj)
    proj_kernel.initialize(lengthscale=initial_ls)
    return proj_kernel


def create_general_rp_poly_kernel(d, degrees, learn_proj=False, weighted=False, kernel_type='RBF',
                                  init_lengthscale_range=(1.0, 1.0), init_mixin_range=(1.0, 1.0),
                                  ski=False, ski_options=None, X=None):
    out_dim = sum(degrees)
    W = torch.cat([rp.gen_rp(d, 1) for _ in range(out_dim)], dim=1).t()
    b = torch.zeros(out_dim)
    projection_module = torch.nn.Linear(d, out_dim, bias=False)
    projection_module.weight = torch.nn.Parameter(W)
    projection_module.bias = torch.nn.Parameter(b)
    if kernel_type == 'RBF':
        kernel = gpytorch.kernels.RBFKernel
        kwargs = dict()
    elif kernel_type == 'Matern':
        kernel = gpytorch.kernels.MaternKernel
        kwargs = dict(nu=1.5)
    else:
        raise ValueError("Unknown kernel type")

    kernel = GeneralizedProjectionKernel(degrees, d, kernel, projection_module, learn_proj, weighted, ski, ski_options,
                                         X=X, **kwargs)
    kernel.initialize(init_mixin_range, init_lengthscale_range)
    return kernel

# TODO: Refactor by wrapping kernel "kinds" such as RP kernel, additive kernel etc. in a class and use inheritance...
def create_rp_kernel(d, k, J, ard=False, activation=None, ski=False,
                     grid_size=None, learn_proj=False, weighted=False, kernel_type='RBF'):
    """Construct a RP kernel object (though not random if learn_proj is true)
    d is dimensionality of data
    k is the dimensionality of the projections
    J is the number of independent RP kernels in a RPKernel object
    ard set to True if each RBF kernel should use ARD
    activation None if no nonlinearity applied after projection. Otherwise, the name of the nonlinearity
    ski set to True computes each sub-kernel by scalable kernel interpolation
    grid_size ignored if ski is False. Otherwise, the size of the grid in each dimension
        * Note that if we project into k dimensions, we have grid_size^d grid points
    learn_proj set to True to learn projection matrix elements
    weighted set to True to learn the linear combination of kernels
    """
    # TODO: implement random initialization
    warnings.warn(DeprecationWarning("create_rp_kernel is deprecated and won't work right from now on."))
    if J < 1:
        raise ValueError("J<1")
    if ard:
        ard_num_dims = k
    else:
        ard_num_dims = None

    kernels = []
    projs = []
    bs = [torch.zeros(k) for _ in range(J)]
    for j in range(J):
        projs.append(rp.gen_rp(d, k))  # d, k just output dimensions of matrix
        if kernel_type == 'RBF':
            kernel = gpytorch.kernels.RBFKernel(ard_num_dims)
        elif kernel_type == 'Matern':
            kernel = gpytorch.kernels.MaternKernel(nu=1.5)
        else:
            raise ValueError("Unknown kernel type")
        if ski:
            kernel = gpytorch.kernels.GridInterpolationKernel(kernel,
                                                              grid_size=grid_size,
                                                              num_dims=k)
        kernels.append(kernel)

    # RP kernel takes a lot of arguments
    return ProjectionKernel(J, k, d, kernels, projs, bs, activation=activation, learn_proj=learn_proj, weighted=weighted)


def create_strictly_additive_kernel(d, weighted=False, kernel_type='RBF', init_lengthscale_range=(1.0, 1.0),
                                    init_mixin_range=(1.0, 1.0), ski=False, ski_options=None, X=None,
                                    memory_efficient=False):
    """Inefficient implementation of a kernel where each dimension has its own RBF subkernel."""

    if kernel_type == 'RBF' and memory_efficient:
        # TODO: account for kernel scaling.
        kernel = MemoryEfficientGamKernel(ard_num_dims=d)
        kernel.initialize(lengthscale=_sample_from_range(d, init_lengthscale_range))
        return kernel
    elif kernel_type == 'RBF':
        kernel = gpytorch.kernels.RBFKernel
        kwargs = dict()
    elif kernel_type == 'Matern':
        kernel = gpytorch.kernels.MaternKernel
        kwargs = dict(nu=3/2)
    else:
        raise ValueError("Unknown kernel type")

    kernel = StrictlyAdditiveKernel(d, kernel, weighted, ski=ski, ski_options=ski_options, X=X, **kwargs)
    kernel.initialize(init_mixin_range, init_lengthscale_range)
    return kernel


def create_additive_kernel(d, groups, weighted=False, kernel_type='RBF', init_lengthscale_range=(1.0, 1.0),
                           init_mixin_range=(1.0, 1.0), ski=False, ski_options=None, X=None):
    if kernel_type == 'RBF':
        kernel = gpytorch.kernels.RBFKernel
        kwargs = dict()
    elif kernel_type == 'Matern':
        kernel = gpytorch.kernels.MaternKernel
        kwargs = dict(nu=3/2)
    else:
        raise ValueError("Unknown kernel type")

    kernel = AdditiveKernel(groups, d, kernel, weighted=weighted, ski=ski, ski_options=ski_options, X=X, **kwargs)
    kernel.initialize(init_mixin_range, init_lengthscale_range)
    return kernel


def create_duvenaud_additive_kernel(d, max_degree):
    kernel = DuvenaudAdditiveKernel(d, max_degree)
    return kernel


def create_multi_additive_kernel(d, max_degree, weighted=False, kernel_type='RBF', init_lengthscale_range=(1.0, 1.0),
                                 init_mixin_range=(1.0, 1.0), ski=False, ski_options=False, X=None):
    if kernel_type == 'RBF':
        kernel = gpytorch.kernels.RBFKernel
        kwargs = dict()
    elif kernel_type == 'Matern':
        kernel = gpytorch.kernels.MaternKernel
        kwargs = dict(nu=3/2)
    else:
        raise ValueError("Unknown kernel type")

    max_degree = min(max_degree, d)
    groups = []
    for deg in range(1, max_degree+1):
        groups.extend(list(set(combinations(list(range(d)), deg))))

    kernel = AdditiveKernel(groups, d, kernel, weighted=weighted, ski=ski, ski_options=ski_options, X=X, **kwargs)
    kernel.initialize(init_mixin_range, init_lengthscale_range)
    return kernel


def create_multi_full_kernel(d, J, ard=False, ski=False, grid_size=None, kernel_type='RBF', init_lengthscale_range=(1.0, 1.0),
                             init_mixin_range=(1.0, 1.0)):
    outputscales = _sample_from_range(J, init_mixin_range)
    total = sum(outputscales)
    outputscales = [o / total for o in outputscales]

    subkernels = []
    for j in range(0,J):
        new_kernel = ScaleKernel(create_full_kernel(d, ard=ard, ski=ski, grid_size=grid_size, kernel_type=kernel_type,
                                                    init_lengthscale_range=init_lengthscale_range))
        new_kernel.initialize(outputscale=outputscales[j])
        subkernels.append(new_kernel)
    return gpytorch.kernels.AdditiveKernel(*subkernels)


def create_pca_kernel(trainX, random_projections=False, k=1, J=1, explained_variance=.99, ski=False, grid_size=None,
                      weighted=False, kernel_type='RBF'):
    # TODO: convert to using PolynomialProjectionKernel
    # TODO: add random initialization
    [n, d] = trainX.shape
    if not random_projections and k != 1:
        raise ValueError("Additive RBF kernel selected but k is not 1.")
    if explained_variance > 1 or explained_variance < 0:
        raise ValueError("Explained variance ratio should be between 0 and 1.")
    if explained_variance == 1:
        n_components = d
    else:
        n_components = explained_variance
    pca = PCA(n_components, copy=False, random_state=123456)
    pca.fit(trainX)
    W = torch.tensor(pca.components_.T, dtype=torch.float)
    D = torch.tensor(pca.explained_variance_, dtype=torch.float)
    if not random_projections:
        J = len(D)

    kernels = []
    projs = []
    bs = [torch.zeros(k) for _ in range(J)]
    for j in range(J):
        if random_projections:
            projs.append(rp.gen_pca_rp(d, k, W.t(), D))
        else:
            projs.append(W[:, j].unsqueeze(1))
        if kernel_type == 'RBF':
            kernel = gpytorch.kernels.RBFKernel()
        elif kernel_type == 'Matern':
            kernel = gpytorch.kernels.MaternKernel(nu=1.5)
        else:
            raise ValueError("Unknown kernel type")
        if ski:
            kernel = gpytorch.kernels.GridInterpolationKernel(kernel, grid_size=grid_size, num_dims=k)
        kernels.append(kernel)
    return ProjectionKernel(J, k, d, kernels, projs, bs, activation=None, learn_proj=False, weighted=weighted)


def create_full_kernel(d, ard=False, ski=False, grid_size=None, kernel_type='RBF', init_lengthscale_range=(1.0, 1.0)):
    """Helper to create an RBF kernel object with these options."""
    if ard:
        ard_num_dims = d
    else:
        ard_num_dims = None
    if kernel_type == 'RBF':
        kernel = gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims)
    elif kernel_type == 'Matern':
        kernel = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=ard_num_dims)
    elif kernel_type == 'InverseMQ':
        kernel = InverseMQKernel(ard_num_dims=ard_num_dims)
    else:
        raise ValueError("Unknown kernel type")

    if ard:
        samples = ard_num_dims
    else:
        samples = 1
    kernel.initialize(lengthscale=_sample_from_range(samples, init_lengthscale_range))

    if ski:
        kernel = GridInterpolationKernel(kernel, num_dims=d,
                                         grid_size=grid_size)
    return kernel


def create_sgpr_kernel(d, ard=False, kernel_type='RBF', inducing_points=800, init_lengthscale_range=(1.0, 1.0),
                       X=None, likelihood=None):
    if ard:
        ard_num_dims = d
    else:
        ard_num_dims = None
    if kernel_type == 'RBF':
        kernel = gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims)
    elif kernel_type == 'Matern':
        kernel = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=ard_num_dims)
    elif kernel_type == 'InverseMQ':
        kernel = InverseMQKernel(ard_num_dims=ard_num_dims)
    else:
        raise ValueError("Unknown kernel type")

    if ard:
        samples = ard_num_dims
    else:
        samples = 1
    kernel.initialize(lengthscale=_sample_from_range(samples, init_lengthscale_range))

    if X is None:
        raise ValueError("X is required")
    if likelihood is None:
        raise ValueError("Likelihood is required")
    kernel = InducingPointKernel(kernel, X[:inducing_points], likelihood)
    return kernel


def create_svi_gp(trainX, kind, **kwargs):
    """Create a SVGP model with a specified kernel.
    rp: if True, use a random projection kernel
    k: dimension of the projections (ignored if rp is False)
    J: number of RP subkernels (ignored if rp is False)
    ard: whether to use ARD in RBF kernels
    activation: passed to create_rp_kernel
    noise_prior: if True, use a box prior over Gaussian observation noise to help with optimization
    ski: if True, use SVI with SKI as in Stochastic Variational Deep Kernel Learning (not implemented yet)
    learn_proj: passed to create_rp_kernel
    additive: if True, (and not RP) use an additive kernel instead of RP or RBF
    weighted: if True, learn the linear combination of sub-kernels (if applicable)
    """
    # TODO: lift to work with PCA, PCA rp, and importantly RP poly and deep rp
    if 'space_proj' in kwargs.keys():
        raise NotImplementedError("SVI GP is not yet implemented with Poly projection kernel and therefore doesn't support equally spaced projections yet.")
    if 'ski' in kwargs.keys():
        raise NotImplementedError("SVI with KISS-GP not implemented")
    if kind == 'deep_rp_poly':
        raise NotImplementedError("SVI with deep projections isn't supported yet.")
    [n, d] = trainX.shape

    # regular Gaussian likelihood for regression problem
    if kwargs.pop('noise_prior'):
        noise_prior_ = gpytorch.priors.SmoothedBoxPrior(1e-4, 10, sigma=0.01)
    else:
        noise_prior_ = None

    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_prior=noise_prior_)
    grid_size = kwargs.pop('grid_size')
    if kind == 'rp':
        kernel = create_rp_kernel(d, **kwargs)
    elif kind == 'additive':
        kernel = create_strictly_additive_kernel(d, ski=False, **kwargs)
    elif kind == 'full':
        kernel = create_full_kernel(d, ski=False, **kwargs)
    else:
        raise NotImplementedError("Unknown kernel structure kind {}".format(kind))
    kernel = ScaleKernel(kernel)

    # Choose initial inducing points as a subset of the data randomly chosen
    if grid_size is None:
        grid_size = min(1000, n)
    choice = torch.multinomial(torch.ones(n), grid_size, replacement=True)
    inducing_points = trainX[choice, :]
    model = SVGPRegressionModel(inducing_points, kernel, likelihood)
    return model, likelihood


def train_svi_gp(trainX, trainY, testX, testY, kind, model_kwargs, train_kwargs):
    """Create and train a SVGP with the given parameters."""

    model, likelihood = create_svi_gp(trainX, kind, **model_kwargs)
    mll = VariationalELBO(likelihood, model, trainY.size(0), combine_terms=False)

    def nmll(*args):
        log_lik, kl_div, log_prior = mll(*args)
        return -(log_lik - kl_div + log_prior)

    optimizer_ = _map_to_optim(train_kwargs.pop('optimizer'))

    likelihood.train()
    model.train()
    trained_epochs = train_to_convergence(model, trainX, trainY, objective=nmll,
                                          optimizer=optimizer_, **train_kwargs)
    model.eval()
    likelihood.eval()
    mll.eval()

    model_metrics = dict()
    model_metrics['trained_epochs'] = trained_epochs
    with torch.no_grad():
        model.train()  # consider prior for evaluation on train dataset
        likelihood.train()
        train_outputs = model(trainX)
        model_metrics['prior_train_nmll'] = nmll(train_outputs, trainY).item()

        model.eval()  # Now consider posterior distributions
        likelihood.eval()
        train_outputs = model(trainX)
        test_outputs = model(testX)
        model_metrics['train_nll'] = -likelihood(train_outputs).log_prob(
            trainY).item()
        model_metrics['test_nll'] = -likelihood(test_outputs).log_prob(
            testY).item()
        model_metrics['train_mse'] = mean_squared_error(train_outputs.mean,
                                                        trainY)

    model_metrics['state_dict_file'] = _save_state_dict(model)

    return model_metrics, test_outputs.mean, model
    # TODO!
    # if ski:
    #     pass


def create_exact_gp(trainX, trainY, kind, devices=('cpu',), **kwargs):
    """Create an exact GP model with a specified kernel.
        rp: if True, use a random projection kernel
        k: dimension of the projections (ignored if rp is False)
        J: number of RP subkernels (ignored if rp is False)
        ard: whether to use ARD in RBF kernels
        activation: passed to create_rp_kernel
        noise_prior: if True, use a box prior over Gaussian observation noise to help with optimization
        ski: if True, use SKI
        grid_ratio: used if grid size is not provided to determine number of inducing points.
        grid_size: the number of grid points in each dimension.
        learn_proj: passed to create_rp_kernel
        additive: if True, (and not RP) use an additive kernel instead of RP or RBF
        """
    [n, d] = trainX.shape
    if kind not in ['full', 'rp', 'strictly_additive', 'additive', 'rp_poly', 'deep_rp_poly',
                    'general_rp_poly', 'multi_full', 'duvenaud_additive', 'additive_rp', 'sgpr']:
        raise ValueError("Unknown kernel structure type {}".format(kind))

    # regular Gaussian likelihood for regression problem
    if kwargs.pop('noise_prior'):
        noise_prior_ = gpytorch.priors.SmoothedBoxPrior(1e-4, 10, sigma=0.01)
    else:
        noise_prior_ = None

    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=noise_prior_)
    likelihood.noise = _sample_from_range(1, kwargs.pop('init_noise_range', [1.0, 1.0]))
    grid_size = kwargs.pop('grid_size', None)
    grid_ratio = kwargs.pop('grid_ratio', None)
    ski = kwargs.get('ski', False)
    if kind == 'full':
        if ski and grid_size is None:
            grid_size = int(grid_ratio * math.pow(n, 1 / d))
        kernel = create_full_kernel(d, grid_size=grid_size, **kwargs)
    elif kind == 'multi_full':
        kernel = create_multi_full_kernel(d, **kwargs)
    elif kind == 'strictly_additive':
        # if ski and grid_size is None:
        #     grid_size = int(grid_ratio * math.pow(n, 1))
        kernel = create_strictly_additive_kernel(d, X=trainX, **kwargs)
    elif kind == 'additive':
        # if ski and grid_size is None:
        #     grid_size = int(grid_ratio * math.pow(n, 1))
        kernel = create_additive_kernel(d, X=trainX, **kwargs)
    elif kind == 'duvenaud_additive':
        kernel = create_duvenaud_additive_kernel(d, **kwargs)
    # elif kind == 'pca':
    #     # TODO: modify to work with PCA
    #     if ski and grid_size is None:
    #         grid_size = int(grid_ratio * math.pow(n, 1))
    #     kernel = create_pca_kernel(trainX,grid_size=grid_size,
    #                                random_projections=False, k=1,
    #                                **kwargs)
    elif kind == 'rp':
        if ski and grid_size is None:
            grid_size = int(grid_ratio * math.pow(n, 1 / kwargs['k']))
        kernel = create_rp_kernel(d, grid_size=grid_size, **kwargs)
    elif kind == 'rp_poly':
        # TODO: check this
        # if ski and grid_size is None:
        #     raise ValueError("I'm pretty sure this is wrong but haven't fixed it yet")
        #     grid_size = int(grid_ratio * math.pow(n, 1 / k))
        kernel = create_rp_poly_kernel(d, X=trainX, **kwargs)
    elif kind == 'deep_rp_poly':
        # if ski and grid_size is None:
        #     raise ValueError("I'm pretty sure this is wrong but haven't fixed it yet")
        #     grid_size = int(grid_ratio * math.pow(n, 1 / k))
        kernel = create_deep_rp_poly_kernel(d, X=trainX, **kwargs)
    elif kind == 'general_rp_poly':
        # if ski:
        #     raise NotImplementedError()
        kernel = create_general_rp_poly_kernel(d, X=trainX, **kwargs)
    elif kind == 'additive_rp':
        kernel = create_additive_rp_kernel(d, **kwargs)
    elif kind == 'sgpr':
        kernel = create_sgpr_kernel(d, X=trainX, likelihood=likelihood, **kwargs)
    # elif kind == 'pca_rp':
    #     # TODO: modify to work with PCA RP
    #     raise NotImplementedError("Apparently not working with PCA RP??")
    #     if grid_size is None:
    #         grid_size = int(grid_ratio * math.pow(n, 1 / k))
    #     kernel = create_pca_kernel(trainX, **kwargs)
    else:
        raise ValueError()

    kernel = gpytorch.kernels.ScaleKernel(kernel)
    if len(devices) > 1:
        kernel = MultiDeviceKernel(kernel, devices, devices[0])
    model = ExactGPModel(trainX, trainY, likelihood, kernel)
    return model, likelihood


def train_ppr_gp(trainX, trainY, testX, testY, model_kwargs, train_kwargs, device='cpu',
                 skip_posterior_variances=False):
    model_kwargs = copy.copy(model_kwargs)
    train_kwargs = copy.copy(train_kwargs)
    d = trainX.shape[-1]
    device = torch.device(device)
    trainX = trainX.to(device)
    trainY = trainY.to(device)
    testX = testX.to(device)
    testY = testY.to(device)

    kernel_type = model_kwargs.pop('kernel_type', 'RBF')
    if kernel_type == 'RBF':
        kernels = [RBFKernel() for _ in range(10)]
    elif kernel_type == 'Matern':
        kernels = [MaternKernel(nu=1.5) for _ in range(10)]
    else:
        raise ValueError("Unknown kernel type")
    if len(model_kwargs) > 0:
        warnings.warn("Not all model kwargs are used: {}".format(list(model_kwargs.keys())))

    optimizer_ = _map_to_optim(train_kwargs.pop('optimizer'))

    model = learn_projections(kernels, trainX, trainY, optimizer=optimizer_, **train_kwargs)
    likelihood = model.likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    model.eval()
    likelihood.eval()
    mll.eval()

    model_metrics = dict()
    with torch.no_grad():
        model.train()  # consider prior for evaluation on train dataset
        likelihood.train()
        train_outputs = model(trainX)
        model_metrics['prior_train_nmll'] = -mll(train_outputs, trainY).item()

        with gpytorch.settings.skip_posterior_variances(skip_posterior_variances):
            model.eval()  # Now consider posterior distributions
            likelihood.eval()
            train_outputs = model(trainX)
            test_outputs = model(testX)
            if not skip_posterior_variances:
                model_metrics['train_nll'] = -likelihood(train_outputs).log_prob(
                    trainY).item()
                model_metrics['test_nll'] = -likelihood(test_outputs).log_prob(
                    testY).item()
            model_metrics['train_mse'] = mean_squared_error(train_outputs.mean,
                                                            trainY)

    model_metrics['state_dict_file'] = _save_state_dict(model)
    return model_metrics, test_outputs.mean.to('cpu'), model


# TODO: raise a warning if somewhat important options are missing.
# TODO: change the key word arguments to model options and rename train_kwargs to train options. This applies to basically all of the functions here.
def train_exact_gp(trainX, trainY, testX, testY, kind, model_kwargs, train_kwargs, devices=('cpu',),
                   skip_posterior_variances=False, skip_random_restart=False, evaluate_on_train=True,
                   output_device=None, record_pred_unc=False):
    """Create and train an exact GP with the given options"""
    model_kwargs = copy.copy(model_kwargs)
    train_kwargs = copy.copy(train_kwargs)
    d = trainX.shape[-1]
    devices = [torch.device(device) for device in devices]
    if output_device is None:
        output_device = devices[0]
    else:
        output_device = torch.device(output_device)
    trainX = trainX.to(output_device)
    trainY = trainY.to(output_device)
    testX = testX.to(output_device)
    testY = testY.to(output_device)

    # replace with value from dataset for convenience
    for k, v in list(model_kwargs.items()):
        if isinstance(v, str) and v == 'd':
            model_kwargs[k] = d

    # Change some options just for initial training with random restarts.
    random_restarts = train_kwargs.pop('random_restarts', 1)
    init_iters = train_kwargs.pop('init_iters', 20)
    optimizer_ = _map_to_optim(train_kwargs.pop('optimizer'))
    rr_check_conv = train_kwargs.pop('rr_check_conv', False)

    initial_train_kwargs = copy.copy(train_kwargs)
    initial_train_kwargs['max_iter'] = init_iters
    initial_train_kwargs['check_conv'] = rr_check_conv
    # initial_train_kwargs['verbose'] = 0  # don't shout about it
    best_model, best_likelihood, best_mll = None, None, None
    best_loss = np.inf

    # TODO: move random restarts code to a train_to_convergence-like function
    if not skip_random_restart:
        # Do some number of random restarts, keeping the best one after a truncated training.
        for restart in range(random_restarts):
            # TODO: log somehow what's happening in the restarts.
            model, likelihood = create_exact_gp(trainX, trainY, kind, devices=devices, **model_kwargs)
            model = model.to(output_device)

            # regular marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            _ = train_to_convergence(model, trainX, trainY, optimizer=optimizer_,
                                     objective=mll, isloss=False, **initial_train_kwargs)
            model.train()
            output = model(trainX)
            loss = -mll(output, trainY).item()
            if loss < best_loss:
                best_loss = loss
                best_model = model
                best_likelihood = likelihood
                best_mll = mll
        model = best_model
        likelihood = best_likelihood
        mll = best_mll
    else:
        model, likelihood = create_exact_gp(trainX, trainY, kind, devices=devices, **model_kwargs)
        model = model.to(output_device)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # fit GP
    trained_epochs = train_to_convergence(model, trainX, trainY, optimizer=optimizer_,
                         objective=mll, isloss=False, **train_kwargs)
    model.eval()
    likelihood.eval()
    mll.eval()

    model_metrics = dict()
    model_metrics['trained_epochs'] = trained_epochs
    with torch.no_grad():
        model.train()  # consider prior for evaluation on train dataset
        likelihood.train()
        train_outputs = model(trainX)
        model_metrics['prior_train_nmll'] = -mll(train_outputs, trainY).item()

        with gpytorch.settings.skip_posterior_variances(skip_posterior_variances):
            model.eval()  # Now consider posterior distributions
            likelihood.eval()
            if evaluate_on_train:
                train_outputs = model(trainX)
                model_metrics['train_mse'] = mean_squared_error(train_outputs.mean, trainY)

            test_outputs = model(testX)
            if not skip_posterior_variances:
                # model_metrics['train_nll'] = -likelihood(train_outputs).log_prob(
                #     trainY).item()
                # model_metrics['test_nll'] = -likelihood(test_outputs).log_prob(
                #     testY).item()
                if evaluate_on_train:
                    model_metrics['train_nll'] = -mll(train_outputs, trainY).item()
                model_metrics['test_nll'] = -mll(test_outputs, testY).item()
                distro = likelihood(test_outputs)
                lower, upper = distro.confidence_region()
                frac = ((testY > lower) * (testY < upper)).to(torch.float).mean().item()
                model_metrics['test_pred_frac_in_cr'] = frac
                if record_pred_unc:
                    model_metrics['test_pred_z_score'] = (testY - distro.mean) / distro.stddev
                    # model_metrics['test_pred_var'] = distro.variance.tolist()
                    # model_metrics['test_pred_mean'] = distro.mean.tolist()

    model_metrics['state_dict_file'] = _save_state_dict(model)
    return model_metrics, test_outputs.mean.to('cpu'), model


def train_compressed_gp(trainX, trainY, testX, testY, model_kwargs, train_kwargs, devices=('cpu',),
                        skip_posterior_variances=False, evaluate_on_train=True,
                        output_device=None, record_pred_unc=False):
    from fitting.sampling import CGPSampler
    d = trainX.shape[-1]
    if len(devices) > 1:
        raise ValueError("CGP not implemented for multi GPUs (yet?)")
    if str(devices[0]) != 'cpu':
        torch.cuda.set_device(torch.device(devices[0]))
    devices = [torch.device(device) for device in devices]
    if output_device is None:
        output_device = devices[0]
    else:
        output_device = torch.device(output_device)
    trainX = trainX.to(output_device)
    trainY = trainY.to(output_device)
    testX = testX.to(output_device)
    testY = testY.to(output_device)

    # Pack all of the kwargs into one object... maybe not the best idea.
    model = CGPSampler(trainX, trainY, **model_kwargs, **train_kwargs)

    model_metrics = dict()
    with torch.no_grad():

        if evaluate_on_train:
            train_outputs = model.pred(trainX)
            model_metrics['train_mse'] = mean_squared_error(train_outputs.mean(), trainY)
        
        test_outputs = model.pred(testX)
        if not skip_posterior_variances:
            if evaluate_on_train:
                model_metrics['train_nll'] = -train_outputs.log_prob(trainY).item()
            model_metrics['test_nll'] = -test_outputs.log_prob(testY).item()
            # TODO: implement confidence region method for model average object.
        model_metrics['sampled_mean_mse'] = mean_squared_error(test_outputs.sample_mean(), testY)
        model_metrics['normal_mean_mse'] = mean_squared_error(test_outputs.mean(), testY)

    # model_metrics['state_dict_file'] = _save_state_dict(model)
    return model_metrics, test_outputs.mean().to('cpu'), model


def train_exact_gp_model_average(trainX, trainY, testX, testY, kind, model_kwargs, train_kwargs,
                                 devices=('cpu',), skip_posterior_variances=False, evaluate_on_train=True,
                                 output_device=None, record_pred_unc=False):
    from fitting.sampling import ModelAverage
    model_kwargs = copy.deepcopy(model_kwargs)
    train_kwargs = copy.deepcopy(train_kwargs)

    if len(devices) > 1:
        raise ValueError("CGP not implemented for multi GPUs (yet?)")
    if str(devices[0]) != 'cpu':
        torch.cuda.set_device(torch.device(devices[0]))
    devices = [torch.device(device) for device in devices]
    if output_device is None:
        output_device = devices[0]
    else:
        output_device = torch.device(output_device)
    trainX = trainX.to(output_device)
    trainY = trainY.to(output_device)
    testX = testX.to(output_device)
    testY = testY.to(output_device)

    predictions, log_mlls = [], []
    varying_params = model_kwargs.pop('varying_params')
    k = list(varying_params.keys())[0]
    for i in range(len(varying_params[k])):
        for k, v in varying_params.items():
            model_kwargs[k] = v[i]
        metrics, pred_mean, model = train_exact_gp(trainX, trainY, testX, testY, kind, model_kwargs, train_kwargs, devices=devices,
                       skip_posterior_variances=skip_posterior_variances, skip_random_restart=True,
                       evaluate_on_train=False)
        log_mlls.append(-metrics['prior_train_nmll'])
        model.eval()
        predictions.append(model(testX))

    test_outputs = ModelAverage(predictions, log_mlls)
    model_metrics = dict()
    with torch.no_grad():
        if not skip_posterior_variances:
            model_metrics['test_nll'] = -test_outputs.log_prob(testY).item()
            # TODO: implement confidence region method for model average object.
        model_metrics['sampled_mean_mse'] = mean_squared_error(test_outputs.sample_mean(), testY)
        model_metrics['normal_mean_mse'] = mean_squared_error(test_outputs.mean(), testY)

    # model_metrics['state_dict_file'] = _save_state_dict(model)
    return model_metrics, test_outputs.mean().to('cpu'), None
