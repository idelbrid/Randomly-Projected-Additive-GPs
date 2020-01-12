import gpytorch
import torch


class ScaledProjectionKernel(gpytorch.kernels.Kernel):
    def __init__(self, projection_module, base_kernel, prescale=False, ard_d=None, learn_proj=False, **kwargs):
        self.has_lengthscale = True
        super(ScaledProjectionKernel, self).__init__(ard_d=ard_d, **kwargs)
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
