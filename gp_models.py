import torch
import gpytorch
import torch.nn.functional as F
from gpytorch.models import ExactGP, AbstractVariationalGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
# from gpytorch.lazy.zero_lazy_tensor import ZeroLazyTensor
from gpytorch.lazy import SumLazyTensor, ConstantMulLazyTensor, lazify, MulLazyTensor
from typing import Optional, Type
import numpy as np
import logging
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def _sample_from_range(num_samples, range_):
    return torch.rand(num_samples) * (range_[1] - range_[0]) + range_[0]


def convert_rp_model_to_additive_model(model, return_proj=True):
    if isinstance(model.covar_module, gpytorch.kernels.ScaleKernel):
        rp_kernel = model.covar_module.base_kernel
    else:
        rp_kernel = model.covar_module
    add_kernel = rp_kernel.to_additive_kernel()
    proj = rp_kernel.projection_module
    if isinstance(model.covar_module, gpytorch.kernels.ScaleKernel):
        add_kernel = gpytorch.kernels.ScaleKernel(add_kernel)
        add_kernel.initialize(outputscale=model.covar_module.outputscale)
    Z = rp_kernel.projection_module(model.train_inputs[0])
    res = AdditiveExactGPModel(Z, model.train_targets, model.likelihood, add_kernel)
    res.mean_module = model.mean_module  # either zero or constant, so doesn't matter if the input is projected.
    if return_proj:
        res = (res, proj)
    return res


class DNN(torch.nn.Module):
    """Note: linear output"""
    def __init__(self, input_dim, output_dim, hidden_layer_sizes, nonlinearity='relu', output_activation='linear',
                 separate_networks=False):
            super(DNN, self).__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.hidden_layer_sizes = hidden_layer_sizes
            self.nonlinearity = nonlinearity
            self.output_activation = output_activation
            self.separate_networks = separate_networks
            if nonlinearity == 'sigmoid':
                self._nonlinearity = F.sigmoid
            elif nonlinearity == 'relu':
                self._nonlinearity = F.relu_
            elif nonlinearity == 'tanh':
                self._nonlinearity = F.tanh
            elif nonlinearity == 'leakyrelu':
                self._nonlinearity = F.leaky_relu_
            else:
                raise ValueError("Unknown nonlinearity")
            if output_activation == 'linear':
                self._output_activation = lambda x: x
            elif output_activation == 'sigmoid':
                self._output_activation = F.sigmoid
            elif output_activation == 'relu':
                self._output_activation = F.relu
            elif output_activation == 'tanh':
                self._output_activation = F.tanh
            else:
                raise ValueError("Unknown output nonlinearity")

            linear_modules = []
            module_list_list = []
            if not separate_networks:
                layer_sizes = [input_dim]+hidden_layer_sizes + [output_dim]
            else:
                layer_sizes = [input_dim]+hidden_layer_sizes + [1]
            for i in range(1, len(layer_sizes)):
                if not separate_networks:
                    linear_modules.append(
                        nn.Linear(layer_sizes[i-1], layer_sizes[i], bias=False)
                    )
                else:
                    toappend = []
                    for j in range(output_dim):
                        toappend.append(nn.Linear(layer_sizes[i-1], layer_sizes[i], bias=False))
                        linear_modules.append(toappend[-1])
                    module_list_list.append(toappend)
            
            self.layers = torch.nn.ModuleList(linear_modules)
            self._layer_list = module_list_list
            
            if not separate_networks:
                for layer in self.layers[:-1]:
                    nn.init.kaiming_uniform_(layer.weight.data, nonlinearity=nonlinearity)
                nn.init.kaiming_uniform_(self.layers[-1].weight.data, nonlinearity=self.output_activation)
            else:
                for layer_set in self._layer_list[:-1]:
                    for layer in layer_set:
                        nn.init.kaiming_uniform_(layer.weight.data, nonlinearity=nonlinearity)
                for layer in self._layer_list[-1]:
                    nn.init.kaiming_uniform_(layer.weight.data, nonlinearity=self.output_activation)
            

    def forward(self, x):
        if not self.separate_networks:
            for i in range(len(self.layers)-1):
                x = self._nonlinearity(self.layers[i](x))
            x = self.layers[-1](x)  # final layer is generally linear
        else:
            output = []
            for j in range(self.output_dim):
                cur_vals = x
                for i in range(len(self._layer_list)-1):
#                     print(cur_vals.shape)
                    cur_vals = self._nonlinearity(self._layer_list[i][j](cur_vals))
#                 print(cur_vals.shape)
                output.append(self._layer_list[-1][j](cur_vals))
            x = torch.cat(output, dim=-1)
#             print(x.shape)
            
        x = self._output_activation(x)
        return x


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class GeneralizedProjectionKernel(gpytorch.kernels.Kernel):
    def __init__(self, component_degrees, d, base_kernel, projection_module,
                 learn_proj=False, weighted=False, ski=False, ski_options=None, X=None, **kernel_kwargs):
        super(GeneralizedProjectionKernel, self).__init__()

        self.learn_proj = learn_proj
        self.projection_module = projection_module
        self.ski = ski
        self.ski_options = ski_options
        self.base_kernel = base_kernel

        # Manually set projection parameters' requires_grad attribute to False if we don't want to learn it.
        if not self.learn_proj:
            for param in self.projection_module.parameters():
                param.requires_grad = False
        else:
            for param in self.projection_module.parameters():
                param.requires_grad = True

        # Special stuff if we're using ski: need to compute projections ahead of time and
        # derive the bounds of the projected data in order to create a static SKI grid.
        if ski and not learn_proj and X is not None:
            # This is a hack b/c X might be on the GPU before we send the model to the GPU
            if X.device != torch.device('cpu'):  # if the data exists on the GPU
                self.projection_module.to(X.device)  # send it to the same device
            proj = self._project(X)
            x_maxs = proj.max(0)[0].tolist()
            x_mins = proj.min(0)[0].tolist()
            spacings = [(xmax - xmin) / (1.0*ski_options['grid_size']-4) for xmin, xmax in zip(x_mins, x_maxs)]
            bounds = [[xmin-2.01*space, xmax+2.01*space] for xmin, xmax, space in zip(x_mins, x_maxs, spacings)]
        else:
            bounds = [None for _ in range(sum(component_degrees))]

        kernels = []
        dim_count = 0
        for i in range(len(component_degrees)):
            product_kernels = []
            for j in range(component_degrees[i]):
                if ski:
                    ad = None
                else:
                    ad = dim_count
                bkernel = base_kernel(active_dims=ad, **kernel_kwargs)
                if ski:
                    bds = None
                    if bounds[dim_count] is not None:
                        bds = [bounds[dim_count]]
                    bkernel = gpytorch.kernels.GridInterpolationKernel(bkernel, active_dims=dim_count,
                                                                       grid_bounds=bds, **ski_options)
                product_kernels.append(bkernel)
                dim_count += 1
            # TODO: re-implement this change when GPyTorch bugfix comes
            # if len(product_kernels) == 1:
            #     product_kernel = product_kernels[0]
            # else:
            product_kernel = gpytorch.kernels.ProductKernel(*product_kernels)
            if weighted:
                product_kernel = gpytorch.kernels.ScaleKernel(product_kernel)
                product_kernel.initialize(outputscale=1/len(component_degrees))
            else:
                product_kernel = gpytorch.kernels.ScaleKernel(product_kernel)
                product_kernel.initialize(outputscale=1 / len(component_degrees))
                product_kernel.raw_outputscale.requires_grad = False
            kernels.append(product_kernel)
        kernel = gpytorch.kernels.AdditiveKernel(*kernels)

        self.kernel = kernel
        self.kernel_kwargs = kernel_kwargs
        self.d = d
        self.component_degrees = component_degrees
        self.J = len(component_degrees)
        self.weighted = weighted
        self.last_x1 = None
        self.cached_projections = None
        self.cache_proj = not learn_proj  # can also manually set to False

    def _project(self, x):
        """Project matrix x with (multiple) random projections"""
        return self.projection_module(x)

    def forward(self, x1, x2, **params):
        """a standard forward with a projection caching mechanism"""
        # Don't cache proj if weights are changing
        if (not self.learn_proj) and self.cache_proj:
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
            return self.kernel(x1_projections)
        else:
            x2_projections = self._project(x2)
            return self.kernel(x1_projections, x2_projections, **params)

    def initialize(self, mixin_range, lengthscale_range):
        mixins = _sample_from_range(len(self.component_degrees), mixin_range)
        mixins = mixins / mixins.sum()
        mixins.requires_grad = False  # todo: double check this does/doesn't make the parameter learnable
        for i, k in enumerate(self.kernel.kernels):
            k.outputscale = mixins[i]
            # TODO: reimplement change when GPyTorch bugfix comes.
            # if self.component_degrees[i] == 1:
            #     subkernels = [k.base_kernel]
            # else:
            subkernels = list(k.base_kernel.kernels)
            for j, kk in enumerate(subkernels):
                ls = _sample_from_range(1, lengthscale_range)
                if not isinstance(kk, gpytorch.kernels.GridInterpolationKernel):
                    kk.lengthscale = ls
                else:
                    kk.base_kernel.lengthscale = ls

    # def eval(self):
    #     if self.ski:
    #         for i, k in enumerate(self.kernel.kernels):
    #             for j, kk in enumerate(k.base_kernel.kernels):
    #                 kk.grid_is_dynamic = True
    #                 print('Setting dynamic true')
    #     return super(GeneralizedProjectionKernel, self).eval()

    def train(self, mode=True):
        if self.ski:
            for i, k in enumerate(self.kernel.kernels):
                if self.component_degrees[i] == 1:
                    subkernels = [k.base_kernel]
                else:
                    subkernels = list(k.base_kernel.kernels)
                for j, kk in enumerate(subkernels):
                    kk.grid_is_dynamic = not mode
                    # print('Setting dynamic {}'.format(not mode))
        return super(GeneralizedProjectionKernel, self).train(mode=mode)

    def to_additive_kernel(self):
        ct = 0
        groups = []
        for d in self.component_degrees:
            g = []
            for _ in range(d):
                g.append(ct)
                ct += 1
            groups.append(g)

        if self.cached_projections is not None:
            Z = self.cached_projections
        else:
            Z = None
        res = AdditiveKernel(groups, self.d, self.base_kernel, self.weighted,
                              self.ski, self.ski_options, X=Z, **self.kernel_kwargs)
        res.kernel = self.kernel  # just use the same object. Probably not a good long-term solution.
        return res

# TODO: make sure this works right
# TODO: implement mixtures of products via kernel powers (log(K) ~ a log(k1) + b log(k2) -> K ~ k1^a * k2^b
class GeneralizedPolynomialProjectionKernel(GeneralizedProjectionKernel):
    def __init__(self, J, k, d, base_kernel, projection_module,
                 learn_proj=False, weighted=False,  ski=False, ski_options=None, X=None, **kernel_kwargs):
        degrees = [k for _ in range(J)]
        super(GeneralizedPolynomialProjectionKernel, self).__init__(degrees, d, base_kernel, projection_module,
                                                                    learn_proj, weighted, ski, ski_options, X=X,
                                                                    **kernel_kwargs)
        self.J = J
        self.k = k


class PolynomialProjectionKernel(GeneralizedPolynomialProjectionKernel):
    def __init__(self, J, k, d, base_kernel, Ws, bs, activation=None,
                 learn_proj=False, weighted=False,  ski=False, ski_options=None, X=None, **kernel_kwargs):
        if activation is not None:
            raise ValueError("activation not supported through the normal projection interface. Use the GeneralPolynomialProjectionKernel instead.")
        projection_module = torch.nn.Linear(d, J*k, bias=False)
        W = torch.nn.Parameter(torch.cat(Ws, dim=1).t())
        b = torch.nn.Parameter(torch.cat(bs, dim=0))
        projection_module.weight = W
        projection_module.bias = b
        super(PolynomialProjectionKernel, self
              ).__init__(J, k, d, base_kernel, projection_module,
                         learn_proj, weighted, ski, ski_options, X=X, **kernel_kwargs)


class AdditiveKernel(GeneralizedProjectionKernel):
    """For convenience"""
    def __init__(self, groups, d, base_kernel, weighted=False, ski=False, ski_options=None, X=None, **kernel_kwargs):
        class GroupFeaturesModule(nn.Module):
            def __init__(self, groups):
                super(GroupFeaturesModule, self).__init__()
                order = []
                for g in groups:
                    order.extend(g)
                self.order = torch.tensor(order)

            def forward(self, x):
                return torch.index_select(x, -1, self.order)

            @property
            def weight(self):
                M = torch.zeros(d, d)
                for i, g in enumerate(self.order):
                    M[i, g] = 1
                return M

        projection_module = GroupFeaturesModule(groups)
        degrees = [len(g) for g in groups]
        super(AdditiveKernel, self).__init__(degrees, d, base_kernel, projection_module,
                                             weighted=weighted, ski=ski, ski_options=ski_options,
                                             X=X, **kernel_kwargs)
        self.groups = groups


class StrictlyAdditiveKernel(AdditiveKernel):
    """For convenience"""
    def __init__(self, d, base_kernel, weighted=False, ski=False, ski_options=None, X=None, **kernel_kwargs):
        # projection_module = Identity()
        groups = [[i] for i in range(d)]
        super(StrictlyAdditiveKernel, self).__init__(groups, d, base_kernel, learn_proj=False,
                                             weighted=weighted, ski=ski, ski_options=ski_options,
                                             X=X, **kernel_kwargs)


class ProjectionKernel(gpytorch.kernels.Kernel):
    def __init__(self, J, k, d, base_kernels, Ws, bs, activation=None,
                 active_dims=None, learn_proj=False, weighted=False, multiplicative=False):
        """

        :param J: Number of independent subkernels
        :param k: Dimension to project into
        :param d: Dimension to project from
        :param base_kernels: Kernel function for each subkernel
        :param Ws: List of projection (weight) matrices
        :param bs: List of offset (bias) vectors (could be 0s)
        :param activation: If not None, apply a nonlinear fxn after projection
        :param active_dims: not used ATM
        :param learn_proj: if true, consider Ws and bs tunable parameters. Otherwise, keep them static.
        :param weighted: if true, learn scale of kernels independently
        """
        super(ProjectionKernel, self).__init__(active_dims=active_dims)
        if not len(base_kernels) == J:
            raise Exception("Number of kernels does not match J")
        if not len(Ws) == J and len(bs) == J:
            raise Exception("Number of weight matrices and/or number of "
                            "bias vectors does not match J")
        if not Ws[0].shape[0] == d:
            raise Exception("Weight matrix 0 number of rows does not match d")
        if not Ws[0].shape[1] == k:
            raise Exception("Weight matrix 0 number of columns does not match k")

        # Register parameters for autograd if learn_proj. Othwerise, don't.
        self.learn_proj = learn_proj
        if self.learn_proj:
            for i in range(len(Ws)):
                self.register_parameter('W_{}'.format(i), torch.nn.Parameter(Ws[i]))
                self.register_parameter('b_{}'.format(i), torch.nn.Parameter(bs[i]))
        else:
            for i in range(len(Ws)):
                self.__setattr__('W_{}'.format(i), Ws[i])
                self.__setattr__('b_{}'.format(i), bs[i])

        self.activation = activation
        # scale each kernel individually if setting "weighted" to true.
        if weighted:
            for i in range(J):
                base_kernels[i] = gpytorch.kernels.ScaleKernel(base_kernels[i])
        self.base_kernels = torch.nn.ModuleList(base_kernels)
        self.d = d
        self.J = J
        self.k = k
        self.weighted = weighted
        self.multiplicative = multiplicative
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
        if not self.learn_proj:
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
        if self.multiplicative:
            res = MulLazyTensor(*base_kernels)
        else:
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


class AdditiveExactGPModel(ExactGPModel):
    def __init__(self, train_x, train_y, likelihood, kernel):
        if isinstance(kernel, gpytorch.kernels.ScaleKernel):
            if not isinstance(kernel.base_kernel, AdditiveKernel):
                raise ValueError("Not an additive kernel.")
        else:
            if not isinstance(kernel, AdditiveKernel):
                raise ValueError("Not an additive kernel.")
        super(AdditiveExactGPModel, self).__init__(train_x, train_y, likelihood, kernel)

    def additive_pred(self, x, group=None):  # Pretty sure this should count as a prediction strategy
        # TODO: implement somehow for RP models as well.
        if isinstance(self.covar_module, gpytorch.kernels.ScaleKernel):
            scale = self.covar_module.outputscale
            add_kernel = self.covar_module.base_kernel
        elif isinstance(self.covar_module,  AdditiveKernel):
            scale = torch.tensor(1.0, dtype=torch.float)
            add_kernel = self.covar_module
        else:
            raise NotImplementedError("Only implemented for Additive kernels and Scale kernel wrappings only")
        train_prior_dist = self.forward(self.train_inputs[0])
        lik_covar_train_train = self.likelihood(train_prior_dist, self.train_inputs[0]).lazy_covariance_matrix.detach()

        # TODO: cache?
        K_inv_y = lik_covar_train_train.inv_matmul(self.train_targets)

        def get_pred(k):
            # is there a better way to handle the scalings?
            test_train_covar = scale * gpytorch.lazy.delazify(k(x, self.train_inputs[0]))
            test_test_covar = scale * gpytorch.lazy.delazify(k(x, x))
            train_test_covar = test_train_covar.transpose(-1, -2)
            pred_mean = test_train_covar.matmul(K_inv_y)
            covar_correction_rhs = lik_covar_train_train.inv_matmul(train_test_covar)
            pred_covar = lazify(torch.addmm(1, test_test_covar, -1, test_train_covar, covar_correction_rhs))
            return gpytorch.distributions.MultivariateNormal(pred_mean, pred_covar, validate_args=False)

        if group is None:
            outputs = []
            for k in add_kernel.kernel.kernels:
                outputs.append(get_pred(k))
            return outputs
        else:
            return get_pred(add_kernel.kernel.kernels[group])

    def get_groups(self):
        if isinstance(self.covar_module, gpytorch.kernels.ScaleKernel):
            add_kernel = self.covar_module.base_kernel
        else:
            add_kernel = self.covar_module
        return add_kernel.groups


class ProjectedAdditiveExactGPModel(ExactGPModel):
    def __init__(self, train_x, train_y, likelihood, kernel):
        if isinstance(kernel, gpytorch.kernels.ScaleKernel):
            if not isinstance(kernel.base_kernel, GeneralizedProjectionKernel):
                raise ValueError("Not an projected additive kernel.")
        else:
            if not isinstance(kernel, GeneralizedProjectionKernel):
                raise ValueError("Not an projected additive kernel.")
        super(ProjectedAdditiveExactGPModel, self).__init__(train_x, train_y, likelihood, kernel)

    def get_corresponding_additive_model(self, return_proj=True):
        return convert_rp_model_to_additive_model(self, return_proj=return_proj)


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


def postprocess_rbf(dist_mat):
    return dist_mat.div_(-2).exp_()


class DuvenaudAdditiveKernel(gpytorch.kernels.Kernel):
    def __init__(self, d, max_degree=None, active_dims=None, **kwargs):
        super(DuvenaudAdditiveKernel, self).__init__(has_lengthscale=True,
                                                     active_dims=active_dims,
                                                     lengthscale_prior=None,
                                                     lengthscale_constraint=None,
                                                     ard_num_dims=d,
                                                     **kwargs
                                                     )
        self.d = d
        if max_degree is None:
            self.max_degree = d
        else:
            self.max_degree = max_degree
        if max_degree > d:
            raise ValueError()

        self.kernels = torch.nn.ModuleList([gpytorch.kernels.RBFKernel() for _ in range(d)])
        self.register_parameter(
            name='raw_outputscale',
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, max_degree))
        )
        outputscale_constraint = gpytorch.constraints.Positive()
        self.register_constraint('raw_outputscale', outputscale_constraint)
        self.outputscale_constraint = outputscale_constraint

    @property
    def outputscale(self):
        return self.raw_outputscale_constraint.transform(self.raw_outputscale)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        x1_ = x1.div(self.lengthscale)  # account for lengthscales first
        x2_ = x2.div(self.lengthscale)
        kern_values = self.covar_dist(x1_, x2_, diag=diag, last_dim_is_batch=True,
                                      square_dist=True,
                                      dist_postprocess_func=postprocess_rbf,
                                      postprocess=True)
        print('Single kern values', kern_values)

        # kern_values = D x n x n
        # last dim is batch, which gets moved up to pos. 1
        # compute scale-less values for each degree
        kvals = torch.range(1, self.max_degree).reshape(-1, 1, 1, 1)
        # kvals = 1 x D (indexes only)
        e_n = torch.ones(self.max_degree+1, *kern_values.shape[1:])  # includes 0
 
        print('degrees', kvals)
        s_k = kern_values.pow(kvals).sum(dim=1)  # should have max_degree # of terms
        print('sum of kernels to degrees', s_k)
        # e_n = R x n x n
        # s_k = R x n x n
        for deg in range(1, self.max_degree+1):
            term = torch.zeros(*e_n.shape[1:])  # 1 x n x n
            for k in range(1, deg+1):
#                 print('k', k)
                # e_n includes zero, s_k does not. Adjust indexing accordingly
                term = term + (-1)**(k-1) * e_n[deg - k] * s_k[k-1]
            e_n[deg] = term / deg
            print(e_n[deg])
        return (self.outputscale.reshape(-1, 1, 1) * e_n[1:]).sum(dim=0)


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


def mean_squared_error(y_pred, y_true):
    """Helper to calculate MSE"""
    return ((y_pred - y_true)**2).mean().item()
