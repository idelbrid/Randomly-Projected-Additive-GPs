import gpytorch
from gpytorch import lazify
from gpytorch.models import ExactGP, AbstractVariationalGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
import torch
from gp_models.kernels import GeneralizedProjectionKernel
from gp_models import CustomAdditiveKernel


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
            if not isinstance(kernel.base_kernel, CustomAdditiveKernel):
                raise ValueError("Not an additive kernel.")
        else:
            if not isinstance(kernel, CustomAdditiveKernel):
                raise ValueError("Not an additive kernel.")
        super(AdditiveExactGPModel, self).__init__(train_x, train_y, likelihood, kernel)

    def additive_pred(self, x, group=None):  # Pretty sure this should count as a prediction strategy
        # TODO: implement somehow for RP models as well.
        if isinstance(self.covar_module, gpytorch.kernels.ScaleKernel):
            scale = self.covar_module.outputscale
            add_kernel = self.covar_module.base_kernel
        elif isinstance(self.covar_module,  CustomAdditiveKernel):
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