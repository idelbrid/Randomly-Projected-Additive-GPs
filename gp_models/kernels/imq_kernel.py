import gpytorch


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
