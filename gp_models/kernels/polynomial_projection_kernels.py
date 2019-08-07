import copy

import gpytorch
import torch
from torch import nn, __init__

import rp
from gp_models.kernels.etc import _sample_from_range


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class GeneralizedProjectionKernel(gpytorch.kernels.Kernel):
    """The complicated base-kernel for polynomial-projection kernels....

    Implements k(x) = k1(p1(x))*k1(p2(x))*...*k1(pj(x)) + ... + k1(pn1(x))*...*k1(pnd(x))
     * We have a single base kernel class base_kernel
     * We have an arbitrary projection module (which may be the identity transformation)
     * In order of the dimensions, the projections are grouped into multiplicative groups by the component_degrees
         key word, e.g. component_degrees=[2, 3] would group projection 1 and 2 together and 3, 4, and 5 together.
     * Each additive component in the covariance decomposition optionally has its own scale, indicated by `weighted`
     * the ski keyword allows each kernel to be approximated by SKI automagically
     * X is used to deterimine the bounds of the projections that SKI will use for its grid.

    Note: This kernel is implemented using AdditiveKernel and ProductKernel, which are _not_ memory efficient.
    """
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
        """Initialize weights/sales and length scales of sub-kernels."""
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
        """Helper function to make an CustomAdditiveKernel from this class."""
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
        res = CustomAdditiveKernel(groups, self.d, self.base_kernel, self.weighted,
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


class CustomAdditiveKernel(GeneralizedProjectionKernel):
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
        super(CustomAdditiveKernel, self).__init__(degrees, d, base_kernel, projection_module,
                                                   weighted=weighted, ski=ski, ski_options=ski_options,
                                                   X=X, **kernel_kwargs)
        self.groups = groups


class StrictlyAdditiveKernel(CustomAdditiveKernel):
    """For convenience"""
    def __init__(self, d, base_kernel, weighted=False, ski=False, ski_options=None, X=None, **kernel_kwargs):
        # projection_module = Identity()
        groups = [[i] for i in range(d)]
        super(StrictlyAdditiveKernel, self).__init__(groups, d, base_kernel, learn_proj=False,
                                             weighted=weighted, ski=ski, ski_options=ski_options,
                                             X=X, **kernel_kwargs)