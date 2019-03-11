import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.lazy.zero_lazy_tensor import ZeroLazyTensor
from typing import Optional, Type
import numpy as np
import logging

#
class RPKernel(gpytorch.kernels.Kernel):
    # TODO: Add unit tests to verify that it works right.
    def __init__(self, J, k, d, base_kernels, Ws, bs, activation=None,
                 active_dims=None):
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

        self.Ws = Ws
        self.bs = bs
        self.activation = activation
        self.base_kernels = base_kernels
        self.d = d
        self.J = J
        self.k = k

    def forward(self, x1, x2, batch_dims=None, **params):
        # res = torch.zeros([x1.size[0], x2.size[0]])
        res = ZeroLazyTensor(x1.shape[0], x2.shape[0])
        for j in range(self.J):
            x1_proj = x1.matmul(self.Ws[j]) + self.bs[j].unsqueeze(0)
            x2_proj = x2.matmul(self.Ws[j]) + self.bs[j].unsqueeze(0)
            if self.activation is None:
                pass
            elif self.activation == 'sigmoid':
                x1_proj = torch.nn.functional.sigmoid(x1_proj)
                x2_proj = torch.nn.functional.sigmoid(x2_proj)
            else:
                raise ValueError("Unimplemented activation")

            res = res + (1/self.J)*self.base_kernels[j](x1_proj, x2_proj, **params)

        return res


class ExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        # train_x = torch.FloatTensor(train_x)
        # train_y = torch.FloatTensor(train_y)

        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        # x = torch.FloatTensor(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# TODO: add logging
def fit_gp_model(gp_model, gp_likelihood, xs, ys,
                 optimizer: Optional[Type]=None, lr=0.1,
                 gp_mll=None, n_epochs=100, verbose=False, patience=20,
                 conv_tol=1e-4, check_conv=True, smooth=True):
    if gp_mll is None:
        gp_mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp_likelihood, gp_model)
    if optimizer is None:
        optimizer = torch.optim.LBFGS

    gp_model.train()
    gp_likelihood.train()

    # instantiating optimizer
    optimizer_ = optimizer(gp_model.parameters(), lr=lr)

    losses = np.zeros((n_epochs,))
    ma = np.zeros((n_epochs,))
    for i in range(n_epochs):

        # Define and pass in closure to work with LBFGS, but also works with
        #     other optimizers like ADAM too.
        def closure():
            optimizer_.zero_grad()
            output = gp_model(xs)
            loss = -gp_mll(output, ys)
            if verbose:
                print(
                        'Iter %d/%d - Loss: %.3f, Noise: %.4f' % (i + 1, n_epochs, loss.item(), gp_likelihood.noise.item()))
            loss.backward()
            return loss
        loss = optimizer_.step(closure).item()
        losses[i] = loss
        ma[i] = losses[i-patience+1:i+1].mean()
        if check_conv and i >= patience:
            if smooth and ma[i-patience] - ma[i] < conv_tol:
                if verbose:
                    print("Reached convergence at {}, MA {} - {} < {}".format(loss, ma[i-patience], ma[i], conv_tol))
                return
            if not smooth and losses[i-patience] - losses[i] < conv_tol:
                if verbose:
                    print("Reached convergence at {}, {} - {} < {}".format(loss, losses[i-patience], loss, conv_tol))
                return
    gp_model.eval()
    gp_likelihood.eval()



class LinearRegressionModel(torch.nn.Module):
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


def fit_linear_model(model, xs, ys, criterion,
                     optimizer: Optional[Type]=None, lr=0.1,
                     n_epochs=100, verbose=False, patience=20,
                     conv_tol=1e-4, check_conv=True):
    optimizer_ = optimizer(model.parameters(), lr=lr)

    model.train()

    losses = np.zeros((n_epochs,))
    for i in range(n_epochs):

        # Define and pass in closure to work with LBFGS, but also works with
        #     other optimizers like ADAM too.
        def closure():
            optimizer_.zero_grad()
            output = model(xs)
            loss = criterion(output, ys)
            if verbose:
                print(
                    'Iter %d/%d - Loss: %.3f' % (i + 1, n_epochs, loss.item()))
            loss.backward()
            return loss
        loss = optimizer_.step(closure).item()
        losses[i] = loss
        if check_conv and i >= patience:
            if losses[i-patience] - losses[i] < conv_tol:
                if verbose:
                    print("Reached convergence, {} - {} < {}".format(losses[i-patience], loss, conv_tol))
                return
    model.eval()
