from torch import nn
import torch
from gp_models.kernels.polynomial_projection_kernels import GeneralizedProjectionKernel


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