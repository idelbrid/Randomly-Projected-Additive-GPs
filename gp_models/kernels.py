import torch
import gpytorch
import torch.nn.functional as F
from gpytorch.lazy import SumLazyTensor, ConstantMulLazyTensor, lazify, MulLazyTensor, delazify
import torch.nn as nn
import rp
import copy


def _sample_from_range(num_samples, range_):
    return torch.rand(num_samples) * (range_[1] - range_[0]) + range_[0]


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
                 learn_proj=False, weighted=False, ski=False, ski_options=None, X=None, lengthscale_prior=None,
                 outputscale_prior=None, **kernel_kwargs):
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
                bkernel = base_kernel(active_dims=ad,
                                      lengthscale_prior=copy.deepcopy(lengthscale_prior),
                                      **kernel_kwargs)
                if ski:
                    bds = None
                    if bounds[dim_count] is not None:
                        bds = [bounds[dim_count]]
                    bkernel = gpytorch.kernels.GridInterpolationKernel(bkernel, active_dims=dim_count,
                                                                       grid_bounds=bds, **ski_options)
                product_kernels.append(bkernel)
                dim_count += 1
            # TODO: re-implement this change when GPyTorch bugfix comes
            if len(product_kernels) == 1:
                product_kernel = product_kernels[0]
                scale_kernel_active_dims = product_kernel.active_dims
            else:
                product_kernel = gpytorch.kernels.ProductKernel(*product_kernels)
                scale_kernel_active_dims = None
            if weighted:
                product_kernel = gpytorch.kernels.ScaleKernel(product_kernel,
                                                              outputscale_prior=copy.deepcopy(outputscale_prior),
                                                              active_dims=scale_kernel_active_dims)
                product_kernel.initialize(outputscale=1/len(component_degrees))
            else:
                product_kernel = gpytorch.kernels.ScaleKernel(product_kernel,
                                                              outputscale_prior=copy.deepcopy(lengthscale_prior),
                                                              active_dims=scale_kernel_active_dims)
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
            return self.kernel(x1_projections, **params)
        else:
            x2_projections = self._project(x2)
            return self.kernel(x1_projections, x2_projections, **params)

    def initialize(self, mixin_range, lengthscale_range):
        mixins = _sample_from_range(len(self.component_degrees), mixin_range)
        mixins = mixins / mixins.sum()
        mixins.requires_grad = False
        for i, k in enumerate(self.kernel.kernels):
            k.outputscale = mixins[i]
            # TODO: reimplement change when GPyTorch bugfix comes.
            if self.component_degrees[i] == 1:
                subkernels = [k.base_kernel]
            else:
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

    @property
    def base_kernels(self):
        kernels = []
        for k in self.kernel.kernels:
            for kk in k.base_kernel.kernels:
                kernels.append(kk)
        return kernels

    @property
    def scale_kernels(self):
        return self.kernel.kernels


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


class RPPolyKernel(PolynomialProjectionKernel):
    def __init__(self, J, k, d, base_kernel, activation=None, learn_proj=False, weighted=False, space_proj=False,
                 ski=False, ski_options=None, X=None, **kernel_kwargs):
        projs = [rp.gen_rp(d, k) for _ in range(J)]
        bs = [torch.zeros(k) for _ in range(J)]

        if space_proj:
            # TODO: If k>1, could implement equal spacing for each set of projs
            newW, _ = rp.space_equally(torch.cat(projs, dim=1).t(), lr=0.1, niter=5000)
            newW.requires_grad = False
            projs = [newW[i:i + 1, :].t() for i in range(J)]
        super(RPPolyKernel, self).__init__(J, k, d, base_kernel, projs, bs, activation=activation,
                                           learn_proj=learn_proj, weighted=weighted, ski=ski, ski_options=ski_options,
                                           X=X, **kernel_kwargs)


class AdditiveKernel(GeneralizedProjectionKernel):
    """For convenience"""
    def __init__(self, groups, d, base_kernel, weighted=False, ski=False, ski_options=None, X=None, **kernel_kwargs):
        class GroupFeaturesModule(nn.Module):
            def __init__(self, groups):
                super(GroupFeaturesModule, self).__init__()
                order = []
                for g in groups:
                    order.extend(g)
                order = torch.tensor(order)
                self.register_buffer('order', order)

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


def postprocess_rbf(dist_mat):
    return dist_mat.div_(-2).exp_()


class DuvenaudAdditiveKernel(gpytorch.kernels.Kernel):
    def __init__(self, d, max_degree=None, active_dims=None, **kwargs):
        self.d = d
        if max_degree is None:
            self.max_degree = d
        else:
            self.max_degree = max_degree
        if max_degree > d:
            self.max_degree = d
        super(DuvenaudAdditiveKernel, self).__init__(has_lengthscale=True,
                                                     active_dims=active_dims,
                                                     lengthscale_prior=None,
                                                     lengthscale_constraint=None,
                                                     ard_num_dims=d,
                                                     **kwargs
                                                     )


        self.kernels = torch.nn.ModuleList([gpytorch.kernels.RBFKernel() for _ in range(d)])
        self.register_parameter(
            name='raw_outputscale',
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, self.max_degree))
        )
        outputscale_constraint = gpytorch.constraints.Positive()
        self.register_constraint('raw_outputscale', outputscale_constraint)
        self.outputscale_constraint = outputscale_constraint
        self.outputscale = [1 / self.max_degree for _ in range(self.max_degree)]


    @property
    def outputscale(self):
        return self.raw_outputscale_constraint.transform(self.raw_outputscale)

    @outputscale.setter
    def outputscale(self, value):
        self._set_outputscale(value)

    def _set_outputscale(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_outputscale)

        self.initialize(raw_outputscale=self.outputscale_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        x1_ = x1.div(self.lengthscale)  # account for lengthscales first
        x2_ = x2.div(self.lengthscale)
        kern_values = self.covar_dist(x1_, x2_, diag=diag, last_dim_is_batch=True,
                                      square_dist=True,
                                      dist_postprocess_func=postprocess_rbf,
                                      postprocess=True)

        # kern_values = D x n x n
        # last dim is batch, which gets moved up to pos. 1
        # compute scale-less values for each degree
        kvals = torch.range(1, self.max_degree, device=kern_values.device).reshape(-1, 1, 1, 1)
        # kvals = 1 x D (indexes only)
        # e_n = torch.ones(self.max_degree+1, *kern_values.shape[1:], device=kern_values.device)  # includes 0
        e_n = torch.zeros(self.max_degree+1, *kern_values.shape[1:], device=kern_values.device)
        e_n[0, :, :] = 1.0
        s_k = kern_values.pow(kvals).sum(dim=1)  # should have max_degree # of terms
        # e_n = R x n x n
        # s_k = R x n x n
        m1 = torch.tensor([-1], dtype=torch.float, device=kern_values.device)
        for deg in range(1, self.max_degree+1):
            # term = torch.zeros(*e_n.shape[1:], device=kern_values.device)  # 1 x n x n
            ks = torch.arange(1, deg+1, device=kern_values.device, dtype=torch.float).reshape(-1, 1, 1)
            kslong = torch.arange(1, deg + 1, device=kern_values.device, dtype=torch.long)
            e_n[deg] = (m1.pow(ks-1) * e_n.index_select(0, deg-kslong) * s_k.index_select(0, kslong-1)).sum(dim=0)
            # for k in range(1, deg+1):
            #     e_n[deg].add_((-1)**(k-1) * e_n[deg - k] * s_k[k-1])
            # e_n[deg].div_(deg)
        return (self.outputscale.reshape(-1, 1, 1) * e_n[1:]).sum(dim=0)


# class LinearRegressionModel(torch.nn.Module):
#     """Currently unused LR model"""
#     def __init__(self, trainX, trainY):
#         super(LinearRegressionModel, self).__init__()
#         [n, d] = trainX.shape
#         [m, k] = trainY.shape
#         if not n == m:
#             raise ValueError
#
#         self.linear = torch.nn.Linear(d, k)
#
#     def forward(self, x):
#         out = self.linear(x)
#         return out


# class ELMModule(torch.nn.Module):
#     """Currently unused "extreme learning machine" model"""
#     def __init__(self, trainX, trainY, A, b, activation='sigmoid'):
#         super(ELMModule, self).__init__()
#         [n, d] = trainX.shape
#         [m, _] = trainY.shape
#         [d_, k] = A.shape
#         if not n == m:
#             raise ValueError
#         if not d == d_:
#             raise ValueError
#         self.linear = torch.nn.Linear(k, 1)
#         self.A = A
#         self.b = b.unsqueeze(0)
#
#     def forward(self, x):
#         hl = x.matmul(self.A)+self.b
#         if self.activation == 'sigmoid':
#             hl = torch.nn.Sigmoid(hl)
#         else:
#             raise ValueError
#         out = self.linear(hl)
#         return out


class ProjectionKernel(gpytorch.kernels.Kernel):
    def __init__(self, projection_module, base_kernel):
        super(ProjectionKernel, self).__init__()
        self.projection_module = projection_module
        self.base_kernel = base_kernel

    def forward(self, x1, x2, **params):
        x1 = self.projection_module(x1)
        x2 = self.projection_module(x2)
        return self.base_kernel(x1, x2, **params)


# TODO: clean up this set of code; a lot of repeated stuff.

class ManualRescaleProjectionKernel(gpytorch.kernels.Kernel):
    def __init__(self, projection_module, base_kernel, prescale=False, ard_num_dims=None, learn_proj=False, **kwargs):
        super(ManualRescaleProjectionKernel, self).__init__(has_lengthscale=True, ard_num_dims=ard_num_dims, **kwargs)
        self.projection_module = projection_module
        self.learn_proj = learn_proj
        if not self.learn_proj:
            for param in self.projection_module.parameters():
                param.requires_grad = False

        self.base_kernel = base_kernel
        for param in self.base_kernel.parameters():
            param.requires_grad = False

        self.prescale = prescale

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        eq = torch.equal(x1, x2)
        if self.prescale:
            x1 = x1.div(self.lengthscale)
        x1 = self.projection_module(x1)
        if not self.prescale:
            x1 = x1.div(self.lengthscale)

        if eq:
            x2 = x1
        else:
            if self.prescale:
                x2 = x2.div(self.lengthscale)
            x2 = self.projection_module(x2)
            if not self.prescale:
                x2 = x2.div(self.lengthscale)
        return self.base_kernel(x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params)


def postprocess_ge_GAM(dist):
    return dist.div_(-2).exp_()

class MemoryEfficientGamKernel(gpytorch.kernels.Kernel):
    def __init__(self, **kwargs):
        super(MemoryEfficientGamKernel, self).__init__(has_lengthscale=True, **kwargs)
        self.covar_dist = GAMFunction()

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        return self.covar_dist.apply(x1, x2, self.lengthscale)

#
# class PseudoAdditiveKernel(gpytorch.kernels.Kernel):
#     def __init__(self, base_kernel, num_dims):
#         super(PseudoAdditiveKernel, self).__init__()
#         self.base_kernel = base_kernel
#         self.num_dims = num_dims
#         self._additive_kernel = gpytorch.kernels.AdditiveStructureKernel(
#             base_kernel, num_dims)
#
#     def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
#         k = self._additive_kernel(x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params)
#         return gpytorch.lazy.delazify(k)


def postprocess_inverse_mq(dist):
    return dist.add_(1).pow_(-1/2)


class InverseMQKernel(gpytorch.kernels.Kernel):
    def __init__(self, **kwargs):
        super(InverseMQKernel, self).__init__(has_lengthscale=True, **kwargs)
        # self.register_parameter('gamma', torch.nn.Parameter(torch.as_tensor(gamma)))

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)
        return self.covar_dist(x1_, x2_, diag=diag, last_dim_is_batch=last_dim_is_batch, square_dist=True,
                               postprocess=True, dist_postprocess_func=postprocess_inverse_mq)
        # return dist.add_(self.gamma**2).pow_(-1/2).mul_(self.gamma**2)


class GAMFunction(torch.autograd.Function):
    """Function to compute sum of RBF kernels with efficient memory usage (Only O(nm) memory)
    The result of forward/backward are n x m matrices, so we can get away with only allocating n x m matrices at a time
        instead of expanding to a d x n x m matrix.
    Does not support batch mode!
    """
    @staticmethod
    def forward(ctx, x1, x2, lengthscale):
        n, d = x1.shape
        m, d2 = x2.shape
        if d2 != d:
            raise ValueError("Dimension mismatch")
        x1_ = x1.div(lengthscale)  # +n x d vector
        x2_ = x2.div(lengthscale)  # +m x d vector
        ctx.save_for_backward(x1, x2, lengthscale)  # maybe have to change?
        kernel = torch.zeros(n, m, dtype=x1_.dtype, device=x1.device)  # use accumulator+loop instead of expansion
        for i in range(d):
            # does cdist still create a new n x m tensor in the graph? Any way to avoid allocating the memory?
            # Should just create temporary n x m tensor and add it to the accumulator.
            with torch.no_grad():
                kernel.add_(torch.cdist(x1_[:, i:i+1], x2_[:, i:i+1]).pow_(2).div_(-2).exp_())
        return kernel

    @staticmethod
    def backward(ctx, grad_output):
        x1, x2, lengthscale = ctx.saved_tensors
        x1_ = x1.div(lengthscale)  # probably could just save the scaled x1/x2 tensors from forward
        x2_ = x2.div(lengthscale)
        n, d = x1.shape
        m, d2 = x2.shape
        num_l = torch.numel(lengthscale)  # support ARD/single lengthscale
        lengthscale_grad = torch.zeros_like(lengthscale)
        x1_grad = torch.zeros_like(x1) if x1.requires_grad else None
        x2_grad = torch.zeros_like(x2) if x2.requires_grad else None

        # Again, use accumulators instead of expansion. Less computationally efficient, but more memory efficient.
        with torch.no_grad():
            for i in range(d):
                sq_dist = torch.cdist(x1_[:, i:i + 1], x2_[:, i:i + 1]).pow_(2)
                K_term = sq_dist.div(-2).exp_()  # one of the kernel summands.
                Delta_K = grad_output * K_term  # reused below
                idx = i if num_l > 1 else 0
                lengthscale_grad[idx] += (Delta_K * sq_dist).sum().div_(lengthscale[idx])

                if x1.requires_grad or x2.requires_grad:
                    signed_diff = x2[:, i].expand(n, -1) - x1[:, i].expand(m, -1).t()
                    if x1.requires_grad:
                        x1_grad[:, i] = (Delta_K * signed_diff).sum(dim=1).div_(lengthscale[idx].pow(2))  # sum over rows/x2s
                    if x2.requires_grad:
                        x2_grad[:, i] = -(Delta_K * signed_diff).sum(dim=0).div_(lengthscale[idx].pow(2))  # sum over columns/x1s

        return x1_grad, x2_grad, lengthscale_grad

