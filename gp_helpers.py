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
    # TODO: Add unit tests to verify that it works right.
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
        toreturn = []
        i = 0
        while hasattr(self, 'W_{}'.format(i)):
            toreturn.append(self.__getattr__('W_{}'.format(i)))
            i += 1
        return toreturn


    @property
    def bs(self):
        toreturn = []
        i = 0
        while hasattr(self, 'b_{}'.format(i)):
            toreturn.append(self.__getattr__('b_{}'.format(i)))
            i += 1
        return toreturn


    def _project(self, x):
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
        if not self.learn_weights:
            if self.last_x1 is not None and torch.equal(x1, self.last_x1):
                x1_projections = self.cached_projections
            else:
                x1_projections = self._project(x1)
                self.last_x1 = x1
                self.cached_projections = x1_projections
        else:
            x1_projections = self._project(x1)
        if torch.equal(x1, x2):
            x2_projections = x1_projections
        else:
            x2_projections = self._project(x2)

        base_kernels = []
        for j in range(self.J):
            k = lazify(self.base_kernels[j](x1_projections[j],
                                            x2_projections[j], **params))
            base_kernels.append(ConstantMulLazyTensor(k, 1/self.J))
        res = SumLazyTensor(*base_kernels)

        return res


class ExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        # train_x = torch.FloatTensor(train_x)
        # train_y = torch.FloatTensor(train_y)

        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SVGPRegressionModel(AbstractVariationalGP):
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

#
# def fit_linear_model(model, xs, ys, criterion,
#                      optimizer: Optional[Type]=None, lr=0.1,
#                      n_epochs=100, verbose=False, patience=20,
#                      conv_tol=1e-4, check_conv=True):
#     optimizer_ = optimizer(model.parameters(), lr=lr)
#
#     model.train()
#
#     losses = np.zeros((n_epochs,))
#     for i in range(n_epochs):
#
#         # Define and pass in closure to work with LBFGS, but also works with
#         #     other optimizers like ADAM too.
#         def closure():
#             optimizer_.zero_grad()
#             output = model(xs)
#             loss = criterion(output, ys)
#             if verbose:
#                 print(
#                     'Iter %d/%d - Loss: %.3f' % (i + 1, n_epochs, loss.item()))
#             loss.backward()
#             return loss
#         loss = optimizer_.step(closure).item()
#         losses[i] = loss
#         if check_conv and i >= patience:
#             if losses[i-patience] - losses[i] < conv_tol:
#                 if verbose:
#                     print("Reached convergence, {} - {} < {}".format(losses[i-patience], loss, conv_tol))
#                 return
#     model.eval()
def mean_squared_error(y_pred: torch.Tensor, y_true: torch.Tensor):
    pass
    return ((y_pred - y_true)**2).mean().item()