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

def create_rp_kernel(d, k, J, ard=False, activation=None, ski=False, grid_size=None):
    if J < 1:
        raise ValueError("J<1")
    if ard:
        ard_num_dims = k
    else:
        ard_num_dims = None

    kernels = []
    projs = []
    bs = [torch.zeros(k)] * J
    for j in range(J):
        projs.append(rp.gen_rp(d, k))  # d, k just output dimensions of matrix
        kernel = gpytorch.kernels.RBFKernel(ard_num_dims)
        if ski:
            kernel = gpytorch.kernels.GridInterpolationKernel(kernel,
                                                              grid_size=grid_size,
                                                              num_dims=k)
        kernels.append(kernel)

    return RPKernel(J, k, d, kernels, projs, bs, activation=activation)


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


def train_svi_gp(trainX, trainY, testX, testY, rp, k=None, J=None, ard=False,
                   activation=None, optimizer='adam', n_epochs=100, lr=0.1,
                   verbose=False, patience=10, smooth=True, noise_prior=False,
                   ski=False, grid_ratio=1, grid_size=None, batch_size=64):
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
        kernel = create_rp_kernel(d, k, J, ard, activation, ski=False)
    else:
        kernel = create_rbf_kernel(d, ard, ski=False)
    kernel = ScaleKernel(kernel)
    if grid_size is None:
        grid_size = 1000
    choice = torch.multinomial(torch.ones(n), grid_size, replacement=True)
    inducing_points = trainX[choice, :]
    model = SVGPRegressionModel(inducing_points, kernel, likelihood)
    mll = VariationalELBO(likelihood, model, trainY.size(0), combine_terms=False)

    def nmll(*args):
        log_lik, kl_div, log_prior = mll(*args)
        return -(log_lik - kl_div + log_prior)

    if optimizer == 'adam':
        optimizer_ = torch.optim.Adam
    elif optimizer == 'sgd':
        optimizer_ = torch.optim.SGD
    elif optimizer == 'lbfgs':
        optimizer_ = torch.optim.LBFGS
    else:
        raise ValueError("Unknown optimizer")

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
    return model_metrics, test_outputs.mean, model
    # TODO!
    # if ski:
    #     pass


def train_exact_gp(trainX, trainY, testX, testY, rp, k=None, J=None, ard=False,
                   activation=None, optimizer='lbfgs', n_epochs=100, lr=0.1,
                   verbose=False, patience=20, smooth=True, noise_prior=False,
                   ski=False, grid_ratio=1, grid_size=None, ):
    [n, d] = trainX.shape

    # regular Gaussian likelihood for regression problem
    if noise_prior:
        noise_prior_ = gpytorch.priors.SmoothedBoxPrior(1e-4, 10, sigma=0.01)
    else:
        noise_prior_ = None

    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_prior=noise_prior_)
    if not rp:
        if grid_size is None:
            grid_size = int(grid_ratio * math.pow(n, 1 / d))
        kernel = create_rbf_kernel(d, ard, ski, grid_size)

    if rp:
        if grid_size is None:
            grid_size = int(grid_ratio * math.pow(n, 1 / k))
        kernel = create_rp_kernel(d, k, J, ard, activation, ski, grid_size)
    kernel = gpytorch.kernels.ScaleKernel(kernel)

    if optimizer == 'lbfgs':
        optimizer_ = torch.optim.LBFGS
    elif optimizer == 'adam':
        optimizer_ = torch.optim.Adam
    elif optimizer == 'sgd':
        optimizer_ = torch.optim.SGD
    else:
        raise ValueError("Unknown optimizer {}".format(optimizer))

    model = ExactGPModel(trainX, trainY, likelihood, kernel)

    # regular marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

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
        model_metrics['train_nll'] = -likelihood(train_outputs).log_prob(
            trainY).item()
        model_metrics['test_nll'] = -likelihood(test_outputs).log_prob(
            testY).item()
        model_metrics['train_mse'] = mean_squared_error(train_outputs.mean,
                                                        trainY)

    return model_metrics, test_outputs.mean


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
