from typing import Optional, Type

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from gp_models.kernels import ProjectionKernel
from gp_models.models import ExactGPModel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import AdditiveKernel

def train_to_convergence(model, xs, ys,
                         optimizer: Optional[Type]=None, lr=0.1, objective=None,
                         max_iter=100, verbose=0, patience=20,
                         conv_tol=1e-4, check_conv=True, smooth=True,
                         isloss=False, batch_size=None):
    """The core optimization routine

    :param model: the model (usually a GPyTorch model, usually an ExactGP model) to fit
    :param xs: training x values
    :param ys: training target values
    :param optimizer: torch optimizer function to use, e.g. torch.optim.Adam
    :param lr: learning rate of the local optimizer
    :param objective: the objective to optimize
    :param max_iter: maximum number of epochs
    :param verbose: if 0, produces no output. If 1, produces update per epoch. If 2, outputs per step
    :param patience: the number of epochs after which, if the objective does not change by the tolerance, we stop
    :param conv_tol: the tolerance to check for convergence with
    :param check_conv: if False, train for exactly max_iter epochs
    :param smooth: If True, use a moving average to smooth the losses over epochs for checking convergence
    :param isloss: If True, the objective is considered a loss and is minimized. Otherwise, obj is maximized.
    :param batch_size: If not None, break the data into mini-batches of size batch_size
    :return:
    """
    if optimizer is None:
        optimizer = torch.optim.LBFGS
    verbose = int(verbose)

    train_dataset = TensorDataset(xs, ys)

    shuffle = not(batch_size is None)
    if batch_size is None:
        batch_size = xs.shape[0]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    model.train()

    # instantiating optimizer
    optimizer_ = optimizer(model.parameters(), lr=lr)

    losses = np.zeros((max_iter,))
    ma = np.zeros((max_iter,))
    for i in range(max_iter):
        total_loss = 0
        for j, (x_batch, y_batch) in enumerate(train_loader):
            # Define and pass in closure to work with LBFGS, but also works with
            #     other optimizers like ADAM too.
            def closure():
                optimizer_.zero_grad()
                output = model(x_batch)
                if isloss:
                    loss = objective(output, y_batch)
                else:
                    loss = -objective(output, y_batch)
                loss.backward()
                return loss
            loss = optimizer_.step(closure).item()
            if verbose > 1:
                print("epoch {}, iter {}, loss {}".format(i, j, loss))
            total_loss = total_loss + loss
        losses[i] = total_loss
        ma[i] = losses[i-patience+1:i+1].mean()
        if verbose > 0:
            print("epoch {}, loss {}".format(i, total_loss))
        if check_conv and i >= patience:
            if smooth and ma[i-patience] - ma[i] < conv_tol:
                if verbose > 0:
                    print("Reached convergence at {}, MA {} - {} < {}".format(total_loss, ma[i-patience], ma[i], conv_tol))
                return i
            if not smooth and losses[i-patience] - losses[i] < conv_tol:
                if verbose > 0:
                    print("Reached convergence at {}, {} - {} < {}".format(total_loss, losses[i-patience], total_loss, conv_tol))
                return i
    return max_iter


def mean_squared_error(y_pred, y_true):
    """Helper to calculate MSE"""
    return ((y_pred - y_true)**2).mean().item()


def learn_projections(base_kernels, xs, ys, max_projections=10,
                      mse_threshold=0.0001, post_fit=False, **optim_kwargs):
    n, d = xs.shape
    residuals = ys
    models = []
    for i in range(max_projections):
        with torch.no_grad():
            coef = torch.pinverse(xs).matmul(residuals)

        base_kernel = base_kernels[i]
        projection = torch.nn.Linear(d, 1, bias=False).to(xs)
        projection.weight.data = coef
        kernel = ProjectionKernel(projection, base_kernel)
        model = ExactGPModel(xs, residuals,
                             GaussianLikelihood(), kernel).to(xs)
        mll = ExactMarginalLogLikelihood(model.likelihood, model).to(xs)
        # mll.train()
        model.train()
        train_to_convergence(model, xs, residuals,
                             objective=mll, **optim_kwargs)

        model.eval()
        models.append(model)
        with torch.no_grad():
            residuals = residuals - model(xs).mean
            mse = (residuals ** 2).mean()
            print(mse.item(), end='; ')
            if mse < mse_threshold:
                break
    print()
    joint_kernel = AdditiveKernel(*[model.covar_module for model in models])
    joint_model = ExactGPModel(xs, ys, GaussianLikelihood(), joint_kernel)

    if post_fit:
        mll = ExactMarginalLogLikelihood(joint_model.likelihood,joint_model)
        train_to_convergence(joint_model, xs, ys, objective=mll, **optim_kwargs)

    return joint_model
