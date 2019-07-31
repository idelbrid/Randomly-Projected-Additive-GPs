import gpytorch
import torch
import numpy as np
import gp_models


def get_lengthscales(kernel):
    if isinstance(kernel, gpytorch.kernels.ScaleKernel):
        return get_lengthscales(kernel.base_kernel)
    elif kernel.has_lengthscale:
        return kernel.lengthscale
    elif isinstance(kernel, gp_models.GeneralizedProjectionKernel):
        ls = []
        for k in kernel.kernel.kernels:
            ls_ = []
            for kk in k.base_kernel.kernels:
                ls_.append(kk.lengthscale.item())
            ls.append(ls_)
        return ls
    else:
        return None


def get_mixins(kernel):
    if isinstance(kernel, gp_models.GeneralizedProjectionKernel):
        mixins = []
        for k in kernel.kernel.kernels:
            mixins.append(k.outputscale.item())
        return mixins
    elif isinstance(kernel, gpytorch.kernels.ScaleKernel):
        return get_mixins(kernel.base_kernel)
    else:
        return None


def get_outputscale(kernel):
    if isinstance(kernel, gpytorch.kernels.ScaleKernel):
        return kernel.outputscale
    else:
        return None


def format_for_str(num_or_list, decimals=3):
    if isinstance(num_or_list, torch.Tensor):
        num_or_list = num_or_list.tolist()
        return format_for_str(num_or_list)
    if isinstance(num_or_list, list):
        return [format_for_str(n) for n in num_or_list]
    elif isinstance(num_or_list, float):
        return np.round(num_or_list, decimals)
    else:
        return ''


@torch.jit.script
def my_cdist(x1, x2):
    """from Jacob Gardner here https://github.com/pytorch/pytorch/issues/15253"""
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
    res = res.clamp_min_(1e-30).sqrt_()
    return res