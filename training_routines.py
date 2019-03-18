import math

from torch import __init__

import rp
from gp_helpers import ExactGPModel, train_to_convergence, RPKernel, \
    LinearRegressionModel, mean_squared_error
import gpytorch
from gpytorch.kernels import ScaleKernel, RBFKernel, GridInterpolationKernel
from gpytorch.mlls import VariationalELBO, VariationalMarginalLogLikelihood
from gp_helpers import SVGPRegressionModel
import torch


def _map_to_optim(optimizer):
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
    d = model.state_dict()
    s = str(d)
    h = hash(s)
    fname = 'model_state_dict_{}.pkl'.format(h)
    torch.save(d, 'models/' + fname)
    return fname


def create_rp_kernel(d, k, J, ard=False, activation=None, ski=False,
                     grid_size=None, learn_weights=False, ):
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
        kernel = gpytorch.kernels.RBFKernel(ard_num_dims)
        if ski:
            kernel = gpytorch.kernels.GridInterpolationKernel(kernel,
                                                              grid_size=grid_size,
                                                              num_dims=k)
        kernels.append(kernel)

    return RPKernel(J, k, d, kernels, projs, bs, activation=activation, learn_weights=learn_weights)


def create_additive_kernel(d, ski=False, grid_size=None):
    k = 1
    J = d
    kernels = []
    projs = []
    bs = [torch.zeros(k) for _ in range(J)]
    for j in range(J):
        P = torch.zeros(d, k)
        P[j, :] = 1
        projs.append(P)
        kernel = gpytorch.kernels.RBFKernel()
        if ski:
            kernel = gpytorch.kernels.GridInterpolationKernel(kernel,
                                                              grid_size=grid_size,
                                                              num_dims=k)
        kernels.append(kernel)
    return RPKernel(J, k, d, kernels, projs, bs, activation=None,
                    learn_weights=False)


def create_rbf_kernel(d, ard=False, ski=False, grid_size=None):
    if ard:
        ard_num_dims = d
    else:
        ard_num_dims = None
    kernel = RBFKernel(ard_num_dims=ard_num_dims)
    if ski:
        kernel = GridInterpolationKernel(kernel, num_dims=d,
                                         grid_size=grid_size)
    return kernel


def create_svi_gp(trainX, rp, k, J, ard, activation, noise_prior, ski,
                  grid_ratio, grid_size, learn_weights=False,
                  additive=False):
    if ski:
        raise NotImplementedError("SVI with KISS-GP not implemented")
    [n, d] = trainX.shape

    # regular Gaussian likelihood for regression problem
    if noise_prior:
        noise_prior_ = gpytorch.priors.SmoothedBoxPrior(1e-4, 10, sigma=0.01)
    else:
        noise_prior_ = None

    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_prior=noise_prior_)

    if rp:
        kernel = create_rp_kernel(d, k, J, ard, activation, ski=False,
                                  learn_weights=learn_weights)
    elif additive:
        kernel = create_additive_kernel(d, ski=False)
    else:
        kernel = create_rbf_kernel(d, ard, ski=False)
    kernel = ScaleKernel(kernel)

    # Choose initial inducing points as a subset of the data randomly chosen
    if grid_size is None:
        grid_size = min(1000, n)
    choice = torch.multinomial(torch.ones(n), grid_size, replacement=True)
    inducing_points = trainX[choice, :]
    model = SVGPRegressionModel(inducing_points, kernel, likelihood)
    return model, likelihood


def train_svi_gp(trainX, trainY, testX, testY, rp, k=None, J=None, ard=False,
                   activation=None, optimizer='adam', n_epochs=100, lr=0.1,
                   verbose=False, patience=10, smooth=True, noise_prior=False,
                   ski=False, grid_ratio=1, grid_size=None, batch_size=64,
                 learn_weights=False, additive=False):

    model, likelihood = create_svi_gp(trainX, rp, k, J, ard, activation,
                                      noise_prior, ski, grid_ratio, grid_size, 
                                      learn_weights=learn_weights,
                                      additive=additive)
    mll = VariationalELBO(likelihood, model, trainY.size(0), combine_terms=False)

    def nmll(*args):
        log_lik, kl_div, log_prior = mll(*args)
        return -(log_lik - kl_div + log_prior)

    optimizer_ = _map_to_optim(optimizer)

    likelihood.train()
    model.train()
    train_to_convergence(model, trainX, trainY, lr=lr, objective=nmll,
                         max_iter=n_epochs, verbose=verbose, patience=patience,
                         smooth=smooth, isloss=True, batch_size=batch_size,
                         optimizer=optimizer_)
    model.eval()
    likelihood.eval()
    mll.eval()

    model_metrics = dict()
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


def create_exact_gp(trainX, trainY, rp, k, J, ard, activation, noise_prior, ski,
                    grid_ratio, grid_size, learn_weights=False, additive=False):
    [n, d] = trainX.shape

    # regular Gaussian likelihood for regression problem
    if noise_prior:
        noise_prior_ = gpytorch.priors.SmoothedBoxPrior(1e-4, 10, sigma=0.01)
    else:
        noise_prior_ = None

    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_prior=noise_prior_)
    if not rp:
        if not additive:
            if grid_size is None:
                grid_size = int(grid_ratio * math.pow(n, 1 / d))
            kernel = create_rbf_kernel(d, ard, ski, grid_size)
        else:
            if grid_size is None:
                grid_size = int(grid_ratio * math.pow(n, 1))
            kernel = create_additive_kernel(d, ski=ski, grid_size=grid_size)
    else:
        if grid_size is None:
            grid_size = int(grid_ratio * math.pow(n, 1 / k))
        kernel = create_rp_kernel(d, k, J, ard, activation, ski, grid_size,
                                  learn_weights=learn_weights)
    kernel = gpytorch.kernels.ScaleKernel(kernel)
    model = ExactGPModel(trainX, trainY, likelihood, kernel)
    return model, likelihood


def train_exact_gp(trainX, trainY, testX, testY, rp, k=None, J=None, ard=False,
                   activation=None, optimizer='lbfgs', n_epochs=100, lr=0.1,
                   verbose=False, patience=20, smooth=True, noise_prior=False,
                   ski=False, grid_ratio=1, grid_size=None, learn_weights=False,
                   additive=False):
    model, likelihood = create_exact_gp(trainX, trainY, rp, k, J, ard,
                                        activation, noise_prior, ski,
                                        grid_ratio, grid_size, 
                                        learn_weights=learn_weights,
                                        additive=additive)

    # regular marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    optimizer_ = _map_to_optim(optimizer)

    # fit GP
    train_to_convergence(model, trainX, trainY, optimizer=optimizer_,
                         isloss=False, lr=lr, max_iter=n_epochs,
                         verbose=verbose, objective=mll, patience=patience,
                         smooth=smooth)
    model.eval()
    likelihood.eval()
    mll.eval()

    model_metrics = dict()
    with torch.no_grad():
        model.train()  # consider prior for evaluation on train dataset
        likelihood.train()
        train_outputs = model(trainX)
        model_metrics['prior_train_nmll'] = -mll(train_outputs, trainY).item()

        model.eval()  # Now consider posterior distributions
        likelihood.eval()
        train_outputs = model(trainX)
        test_outputs = model(testX)
        # model_metrics['train_nll'] = -likelihood(train_outputs).log_prob(
        #     trainY).item()
        # model_metrics['test_nll'] = -likelihood(test_outputs).log_prob(
        #     testY).item()
        model_metrics['train_mse'] = mean_squared_error(train_outputs.mean,
                                                        trainY)

    model_metrics['state_dict_file'] = _save_state_dict(model)
    return model_metrics, test_outputs.mean, model


def train_SE_gp(trainX, trainY, testX, testY, **kwargs):
    return train_exact_gp(trainX, trainY, testX, testY, rp=False, **kwargs)


def train_additive_rp_gp(trainX, trainY, testX, testY, **kwargs):
    return train_exact_gp(trainX, trainY, testX, testY, rp=True, **kwargs)


def train_lr(trainX, trainY, testX, testY, optimizer='lbfgs', **kwargs):
    model = LinearRegressionModel(trainX, trainY)
    loss = torch.nn.MSELoss()

    if optimizer == 'lbfgs':
        optimizer_ = torch.optim.LBFGS
    elif optimizer == 'adam':
        optimizer_ = torch.optim.Adam
    elif optimizer == 'sgd':
        optimizer_ = torch.optim.SGD
    else:
        raise ValueError("Unknown optimizer {}".format(optimizer))
    train_to_convergence(model, trainX, trainY, optimizer_, objective=loss,
                         isloss=True, **kwargs)

    model.eval()
    model_metrics = dict()
    ypred = model(testX)

    return model_metrics, ypred
