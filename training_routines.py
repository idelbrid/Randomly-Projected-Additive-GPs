import math

from torch import __init__

import rp
from gp_models import ExactGPModel, train_to_convergence, ProjectionKernel, \
    LinearRegressionModel, mean_squared_error, PolynomialProjectionKernel, DNN,\
    GeneralizedPolynomialProjectionKernel, GeneralizedProjectionKernel
import gpytorch
from gpytorch.kernels import ScaleKernel, RBFKernel, GridInterpolationKernel
from gpytorch.mlls import VariationalELBO, VariationalMarginalLogLikelihood
from gp_models import SVGPRegressionModel
import torch
from sklearn.decomposition import PCA
import warnings
import copy
import numpy as np

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
    torch.save(d, 'models/' + fname)
    return fname


def _sample_from_range(num_samples, range_):
    return torch.rand(num_samples) * (range_[1] - range_[0]) + range_[0]


def create_deep_rp_poly_kernel(d, k, J, projection_architecture, projection_kwargs, learn_proj=False,
                               weighted=False, kernel_type='RBF', init_mixin_range=(1.0, 1.0),
                               init_lengthscale_range=(1.0, 1.0), ski=False, ski_options=None,
                               X=None):
    if projection_architecture == 'dnn':
        module = DNN(d, k*J, **projection_kwargs)
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

    kernel = GeneralizedPolynomialProjectionKernel(J, k, d, kernel, module,
                                                 learn_proj=learn_proj,
                                                 weighted=weighted, ski=ski, ski_options=ski_options, X=X,
                                                   **kwargs)
    kernel.initialize(init_mixin_range, init_lengthscale_range)
    return kernel


def create_rp_poly_kernel(d, k, J, activation=None,
                          learn_proj=False, weighted=False, kernel_type='RBF',
                          space_proj=False, init_mixin_range=(1.0, 1.0), init_lengthscale_range=(1.0, 1.0),
                          ski=False, ski_options=None, X=None,
                          ):
    projs = [rp.gen_rp(d, k) for _ in range(J)]
    bs = [torch.zeros(k) for _ in range(J)]

    if space_proj:
        # TODO: If k>1, could implement equal spacing for each set of projs
        newW, _ = rp.space_equally(torch.cat(projs,dim=1).t(), lr=0.1, niter=5000)
        newW.requires_grad = False
        projs = [newW[i:i+1, :].t() for i in range (J)]

    if kernel_type == 'RBF':
        kernel = gpytorch.kernels.RBFKernel
        kwargs = dict()
    elif kernel_type == 'Matern':
        kernel = gpytorch.kernels.MaternKernel
        kwargs = dict(nu=1.5)
    else:
        raise ValueError("Unknown kernel type")

    kernel = PolynomialProjectionKernel(J, k, d, kernel, projs, bs, activation=activation, learn_proj=learn_proj,
                                        weighted=weighted, ski=ski, ski_options=ski_options, X=X, **kwargs)
    kernel.initialize(init_mixin_range, init_lengthscale_range)
    return kernel


def create_general_rp_poly_kernel(d, degrees, learn_proj=False, weighted=False, kernel_type='RBF',
                                  init_lengthscale_range=(1.0, 1.0), init_mixin_range=(1.0, 1.0),
                                  ski=False, ski_options=None, X=None):
    out_dim = sum(degrees)
    W = torch.cat([rp.gen_rp(d, 1) for _ in range(out_dim)], dim=1).t()
    b = torch.zeros(out_dim)
    projection_module = torch.nn.Linear(d, out_dim)
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


def create_additive_kernel(d, ski=False, grid_size=None, weighted=False, kernel_type='RBF'):
    """Inefficient implementation of a kernel where each dimension has its own RBF subkernel."""
    # TODO: convert to using PolynomialProjectionKernel
    # TODO: add random initialization
    k = 1
    J = d
    kernels = []
    projs = []
    bs = [torch.zeros(k) for _ in range(J)]
    for j in range(J):
        P = torch.zeros(d, k)
        P[j, :] = 1
        projs.append(P)
        if kernel_type == 'RBF':
            kernel = gpytorch.kernels.RBFKernel()
        elif kernel_type == 'Matern':
            kernel = gpytorch.kernels.MaternKernel(nu=1.5)
        else:
            raise ValueError("Unknown kernel type")
        if ski:
            kernel = gpytorch.kernels.GridInterpolationKernel(kernel,
                                                              grid_size=grid_size,
                                                              num_dims=k)
        kernels.append(kernel)
    return ProjectionKernel(J, k, d, kernels, projs, bs, activation=None,
                            learn_proj=False, weighted=weighted)


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
        kernel = gpytorch.kernels.RBFKernel(ard_num_dims)
    elif kernel_type == 'Matern':
        kernel = gpytorch.kernels.MaternKernel(nu=1.5)
    else:
        raise ValueError("Unknown kernel type")

    kernel.lengthscale = _sample_from_range(1, init_lengthscale_range)

    if ski:
        kernel = GridInterpolationKernel(kernel, num_dims=d,
                                         grid_size=grid_size)
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
        kernel = create_additive_kernel(d, ski=False, **kwargs)
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


def create_exact_gp(trainX, trainY, kind, **kwargs):
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
    if kind not in ['full', 'rp', 'additive', 'pca', 'pca_rp', 'rp_poly', 'deep_rp_poly', 'general_rp_poly']:
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
    elif kind == 'additive':
        if ski and grid_size is None:
            grid_size = int(grid_ratio * math.pow(n, 1))
        kernel = create_additive_kernel(d, grid_size=grid_size, **kwargs)
    elif kind == 'pca':
        # TODO: modify to work with PCA
        if ski and grid_size is None:
            grid_size = int(grid_ratio * math.pow(n, 1))
        kernel = create_pca_kernel(trainX,grid_size=grid_size,
                                   random_projections=False, k=1,
                                   **kwargs)
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
    elif kind == 'pca_rp':
        # TODO: modify to work with PCA RP
        raise NotImplementedError("Apparently not working with PCA RP??")
        if grid_size is None:
            grid_size = int(grid_ratio * math.pow(n, 1 / k))
        kernel = create_pca_kernel(trainX, **kwargs)
    else:
        raise ValueError()

    kernel = gpytorch.kernels.ScaleKernel(kernel)
    model = ExactGPModel(trainX, trainY, likelihood, kernel)
    return model, likelihood


# TODO: raise a warning if somewhat important options are missing.
# TODO: change the key word arguments to model options and rename train_kwargs to train options. This applies to basically all of the functions here.
def train_exact_gp(trainX, trainY, testX, testY, kind, model_kwargs, train_kwargs, device='cpu'):
    """Create and train an exact GP with the given options"""
    model_kwargs = copy.copy(model_kwargs)
    train_kwargs = copy.copy(train_kwargs)

    device = torch.device(device)
    trainX = trainX.to(device)
    trainY = trainY.to(device)
    testX = testX.to(device)
    testY = testY.to(device)

    # Change some options just for initial training with random restarts.
    random_restarts = train_kwargs.pop('random_restarts', 1)
    init_iters = train_kwargs.pop('init_iters', 20)
    optimizer_ = _map_to_optim(train_kwargs.pop('optimizer'))

    initial_train_kwargs = copy.copy(train_kwargs)
    initial_train_kwargs['max_iter'] = init_iters
    initial_train_kwargs['check_conv'] = False
    initial_train_kwargs['verbose'] = 0  # don't shout about it
    best_model, best_likelihood, best_mll = None, None, None
    best_loss = np.inf

    # TODO: move random restarts code to a train_to_convergence-like function
    # Do some number of random restarts, keeping the best one after a truncated training.
    for restart in range(random_restarts):
        # TODO: log somehow what's happening in the restarts.
        model, likelihood = create_exact_gp(trainX, trainY, kind, **model_kwargs)
        model = model.to(device)

        # regular marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        _ = train_to_convergence(model, trainX, trainY, optimizer=optimizer_,
                                 objective=mll, isloss=False, **initial_train_kwargs)
        model.eval()
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
    return model_metrics, test_outputs.mean.to('cpu'), model

