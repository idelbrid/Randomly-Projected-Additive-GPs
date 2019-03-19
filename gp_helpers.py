import torch
import gpytorch
from gpytorch.models import ExactGP, AbstractVariationalGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
# from gpytorch.lazy.zero_lazy_tensor import ZeroLazyTensor
from gpytorch.lazy import SumLazyTensor, ConstantMulLazyTensor, lazify
from typing import Optional, Type
import numpy as np
import logging
from torch.utils.data import TensorDataset, DataLoader


class RPKernel(gpytorch.kernels.Kernel):
    # TODO: Add unit tests to test this better.
    def __init__(self, J, k, d, base_kernels, Ws, bs, activation=None,
                 active_dims=None, learn_weights=False):
        super(RPKernel, self).__init__(active_dims=active_dims)
        if not len(base_kernels) == J:
            raise Exception("Number of kernels does not match J")
        if not len(Ws) == J and len(bs) == J:
            raise Exception("Number of weight matrices and/or number of "
                            "bias vectors does not match J")
        if not Ws[0].shape[0] == d:
            raise Exception("Weight matrix 0 number of rows does not match d")
        if not Ws[0].shape[1] == k:
            raise Exception("Weight matrix 0 number of columns does not match d")

        # Register parameters for autograd if learn_weights. Othwerise, don't.
        self.learn_weights = learn_weights
        if self.learn_weights:
            for i in range(len(Ws)):
                self.register_parameter('W_{}'.format(i), torch.nn.Parameter(Ws[i]))
                self.register_parameter('b_{}'.format(i), torch.nn.Parameter(bs[i]))
        else:
            for i in range(len(Ws)):
                self.__setattr__('W_{}'.format(i), Ws[i])
                self.__setattr__('b_{}'.format(i), bs[i])

        self.activation = activation
        self.base_kernels = base_kernels
        self.d = d
        self.J = J
        self.k = k
        self.last_x1 = None
        self.cached_projections = None


    @property
    def Ws(self):
        """Convenience for getting the individual parameters/attributes."""
        toreturn = []
        i = 0
        while hasattr(self, 'W_{}'.format(i)):
            toreturn.append(self.__getattr__('W_{}'.format(i)))
            i += 1
        return toreturn

    @property
    def bs(self):
        """Convenience for getting individual parameters/attributes."""
        toreturn = []
        i = 0
        while hasattr(self, 'b_{}'.format(i)):
            toreturn.append(self.__getattr__('b_{}'.format(i)))
            i += 1
        return toreturn

    def _project(self, x):
        """Project matrix x with (multiple) random projections"""
        projections = []
        for j in range(self.J):
            linear_projection = x.matmul(self.Ws[j]) + self.bs[j].unsqueeze(0)
            if self.activation is None:
                projections.append(linear_projection)
            elif self.activation == 'sigmoid':
                projections.append(
                    torch.nn.functional.sigmoid(linear_projection)
                )
            else:
                raise ValueError("Unimplemented activation function")

        return projections

    def forward(self, x1, x2, **params):
        # Don't cache proj if weights are changing
        if not self.learn_weights:
            # if x1 is the same, don't re-project
            if self.last_x1 is not None and torch.equal(x1, self.last_x1):
                x1_projections = self.cached_projections
            else:
                x1_projections = self._project(x1)
                self.last_x1 = x1
                self.cached_projections = x1_projections
        else:
            x1_projections = self._project(x1)
        # if x2 is x1, which is common, don't re-project.
        if torch.equal(x1, x2):
            x2_projections = x1_projections
        else:
            x2_projections = self._project(x2)

        base_kernels = []
        for j in range(self.J):
            # lazy tensor wrapper for kernel evaluation
            k = lazify(self.base_kernels[j](x1_projections[j],
                                            x2_projections[j], **params))
            # Multiply each kernel by constant 1/J
            base_kernels.append(ConstantMulLazyTensor(k, 1/self.J))
        # Sum lazy tensor for efficient computations...
        res = SumLazyTensor(*base_kernels)

        return res


class ExactGPModel(ExactGP):
    """Basic exact GP model with const mean and a provided kernel"""
    def __init__(self, train_x, train_y, likelihood, kernel):


        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SVGPRegressionModel(AbstractVariationalGP):
    """SVGP with provided kernel, Cholesky variational prior, and provided inducing points."""
    def __init__(self, inducing_points, kernel, likelihood):
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0))
        variational_strategy = VariationalStrategy(self,
                                                   inducing_points,
                                                   variational_distribution,
                                                   learn_inducing_locations=True)
        super(SVGPRegressionModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel
        self.likelihood = likelihood

    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred


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
                return
            if not smooth and losses[i-patience] - losses[i] < conv_tol:
                if verbose > 0:
                    print("Reached convergence at {}, {} - {} < {}".format(total_loss, losses[i-patience], total_loss, conv_tol))
                return


class LinearRegressionModel(torch.nn.Module):
    """Currently unused LR model"""
    def __init__(self, trainX, trainY):
        super(LinearRegressionModel, self).__init__()
        [n, d] = trainX.shape
        [m, k] = trainY.shape
        if not n == m:
            raise ValueError

        self.linear = torch.nn.Linear(d, k)

    def forward(self, x):
        out = self.linear(x)
        return out


class ELMModule(torch.nn.Module):
    """Currently unused "extreme learning machine" model"""
    def __init__(self, trainX, trainY, A, b, activation='sigmoid'):
        super(ELMModule, self).__init__()
        [n, d] = trainX.shape
        [m, _] = trainY.shape
        [d_, k] = A.shape
        if not n == m:
            raise ValueError
        if not d == d_:
            raise ValueError
        self.linear = torch.nn.Linear(k, 1)
        self.A = A
        self.b = b.unsqueeze(0)

    def forward(self, x):
        hl = x.matmul(self.A)+self.b
        if self.activation == 'sigmoid':
            hl = torch.nn.Sigmoid(hl)
        else:
            raise ValueError
        out = self.linear(hl)
        return out


def mean_squared_error(y_pred: torch.Tensor, y_true: torch.Tensor):
    """Helper to calculate MSE"""
    return ((y_pred - y_true)**2).mean().item()