import gpytorch
import torch


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
                # kernel.add_(torch.cdist(x1_[:, i:i+1], x2_[:, i:i+1]).pow_(2).div_(-2).exp_())
                # kernel.add_((x1_[:, i].expand(m, -1).t() - x2_[:,i].expand(n, -1)).pow_(2).div_(-2).exp_())
                # The cdist implementation is dramatically slower! But the above is too data hungry somehow?
                #   it must be due to the double 'expand's. The below is almost as fast and saves memory.
                kernel.add_((x1_[:, i].view(n, 1) - x2_[:, i].expand(n, -1)).pow_(2).div_(-2).exp_())
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
                sq_dist = (x2_[:, i].expand(n,-1) - x1_[:, i].view(n, 1)).pow_(2)
                # sq_dist = torch.cdist(x1_[:, i:i + 1], x2_[:, i:i + 1]).pow_(2)
                Delta_K = sq_dist.div(-2).exp_().mul_(grad_output)  # Reused below.
                idx = i if num_l > 1 else 0
                lengthscale_grad[...,idx].add_(sq_dist.mul_(Delta_K).sum().div(lengthscale[..., idx]))

                if x1.requires_grad or x2.requires_grad:
                    Delta_K_diff = (x2_[:, i].expand(n, -1) - x1_[:, i].view(n, 1)).mul_(Delta_K)
                    if x1.requires_grad:
                        x1_grad[:, i] = Delta_K_diff.sum(dim=1).div_(lengthscale[idx])  # sum over rows/x2s
                    if x2.requires_grad:
                        x2_grad[:, i] = -Delta_K_diff.sum(dim=0).div_(lengthscale[idx])  # sum over columns/x1s
        return x1_grad, x2_grad, lengthscale_grad


class MemoryEfficientGamKernel(gpytorch.kernels.Kernel):
    def __init__(self, **kwargs):
        self.has_lengthscale = True
        super(MemoryEfficientGamKernel, self).__init__(has_lengthscale=True, **kwargs)
        self.covar_dist = GAMFunction()

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        return self.covar_dist.apply(x1, x2, self.lengthscale)
