from unittest import TestCase
from gp_helpers import ProjectionKernel
from rp import gen_rp
from rp_experiments import load_dataset
import torch
import gpytorch
import numpy as np


fake_data = 10+torch.randn(100, 100)*30   # 100 features, 100 points
real_data = torch.Tensor(load_dataset('breastcancer').iloc[:, 1:-1].values)
real_n, real_d = real_data.shape


def pairwise_distance(x1, x2):
    n, d = x1.shape
    dists = torch.zeros(n, n)
    for i in range(n):
        dists[i, :] = (x1[i, :] - x2).pow(2).sum(dim=1).sqrt()
    return dists


# These are lazy unit tests copied and pasted.
# TODO: refactor tests.
class TestRPGenerator(TestCase):
    def test_gen_gaussian(self):
        W = gen_rp(100, 1000, dist='gaussian')
        fake_dists = pairwise_distance(fake_data, fake_data)
        proj = fake_data.matmul(W)
        dists = pairwise_distance(proj, proj)
        self.assertLess((fake_dists - dists).abs().mean() / fake_dists.abs().mean(), 0.1)  # no strict guarantee...

        # ex_norm = fake_data[0, :].norm()
        # proj_norm = proj[0, :].norm()
        # self.assertLess((ex_norm - proj_norm).abs(), 0.01)

        W = gen_rp(real_d, 1000, dist='gaussian')
        real_dists = pairwise_distance(real_data, real_data)
        proj = real_data.matmul(W)
        dists = pairwise_distance(proj, proj)
        self.assertLess((real_dists - dists).abs().mean() / fake_dists.abs().mean(), 0.1)  # no strict guarantee...

    def test_gen_spherical(self):
        W = gen_rp(100, 1000, dist='sphere')
        fake_dists = pairwise_distance(fake_data, fake_data)
        proj = fake_data.matmul(W)
        dists = pairwise_distance(proj, proj)
        self.assertLess((fake_dists - dists).abs().mean() / fake_dists.abs().mean(), 0.1)  # no strict guarantee...

        self.assertAlmostEqual(W[:, 0].norm(), W[:, 1].norm())

        W = gen_rp(real_d, 1000, dist='sphere')
        real_dists = pairwise_distance(real_data, real_data)
        proj = real_data.matmul(W)
        dists = pairwise_distance(proj, proj)
        self.assertLess((real_dists - dists).abs().mean() / fake_dists.abs().mean(), 0.1)  # no strict guarantee...

    def test_gen_bernoulli(self):
        W = gen_rp(100, 1000, dist='bernoulli')
        fake_dists = pairwise_distance(fake_data, fake_data)
        proj = fake_data.matmul(W)
        dists = pairwise_distance(proj, proj)
        self.assertLess((fake_dists - dists).abs().mean() / fake_dists.abs().mean(), 0.1)  # no strict guarantee...

        W = gen_rp(real_d, 1000, dist='bernoulli')
        real_dists = pairwise_distance(real_data, real_data)
        proj = real_data.matmul(W)
        dists = pairwise_distance(proj, proj)
        self.assertLess((real_dists - dists).abs().mean() / fake_dists.abs().mean(), 0.1)  # no strict guarantee...

    def test_gen_uniform(self):
        W = gen_rp(100, 1000, dist='uniform')
        fake_dists = pairwise_distance(fake_data, fake_data)
        proj = fake_data.matmul(W)
        dists = pairwise_distance(proj, proj)
        self.assertLess((fake_dists - dists).abs().mean() / fake_dists.abs().mean(), 0.1)  # no strict guarantee...

        W = gen_rp(real_d, 1000, dist='uniform')
        real_dists = pairwise_distance(real_data, real_data)
        proj = real_data.matmul(W)
        dists = pairwise_distance(proj, proj)
        self.assertLess((real_dists - dists).abs().mean() / fake_dists.abs().mean(), 0.1)  # no strict guarantee...


class TestProjectionKernel(TestCase):
    def setUp(self):
        self.k = 2
        self.d = real_d
        self.J = 3
        self.Ws = [torch.eye(real_d, 2) for _ in range(self.J)]
        self.bs = [torch.zeros(2) for _ in range(self.J)]
        self.base_kernels = [gpytorch.kernels.RBFKernel() for _ in range(self.J)]

    def test_init(self):
        kernel = ProjectionKernel(self.J, self.k, self.d, self.base_kernels, self.Ws, self.bs)
        param_dict = dict()
        for name, param in kernel.named_parameters():
            param_dict[name] = param
        self.assertEqual(len(param_dict), self.J)
        self.assertEqual(param_dict['base_kernels.1.raw_lengthscale'], 0)
        self.assertIn('base_kernels.0.raw_lengthscale', param_dict.keys())
        self.assertIn('base_kernels.2.raw_lengthscale', param_dict.keys())

    def test_forward(self):
        kernel = ProjectionKernel(self.J, self.k, self.d, self.base_kernels, self.Ws, self.bs)
        output = kernel(real_data, real_data)
        self.assertIsInstance(output, gpytorch.lazy.LazyEvaluatedKernelTensor)
        evl = output.evaluate_kernel()
        self.assertIsInstance(evl, gpytorch.lazy.SumLazyTensor)
        self.assertIsInstance(evl.lazy_tensors[0], gpytorch.lazy.ConstantMulLazyTensor)

        K_proj = output.evaluate()

        k1 = gpytorch.kernels.RBFKernel()
        k2 = gpytorch.kernels.RBFKernel()
        k3 = gpytorch.kernels.RBFKernel()
        equivalent = k1(real_data[:, :2]) + k2(real_data[:, :2]) + k3(real_data[:, :2])
        K_eq = equivalent.evaluate() / 3

        indices = np.random.choice(np.array(real_n), size=10)
        for i in indices:
            self.assertAlmostEqual(K_proj[i, 0].detach().numpy(), K_eq[i, 0].detach().numpy(), places=5)

    def test_caching(self):
        kernel = ProjectionKernel(self.J, self.k, self.d, self.base_kernels, self.Ws, self.bs)
        self.assertIsNone(kernel.last_x1)
        self.assertIsNone(kernel.cached_projections)
        _ = kernel(real_data, real_data).evaluate_kernel()
        self.assertIsNotNone(kernel.last_x1)
        self.assertIsNotNone(kernel.cached_projections)
        new = kernel(real_data[:10], real_data[:10]).evaluate()
        self.assertEqual(kernel.last_x1.numel(), real_data[:10].numel())
        self.assertEqual(new.numel(), 100)

    def test_activation(self):
        kernel = ProjectionKernel(self.J, self.k, self.d, self.base_kernels, self.Ws, self.bs, activation='sigmoid')
        output = kernel(real_data, real_data)
        K_proj = output.evaluate()

        k1 = gpytorch.kernels.RBFKernel()
        k2 = gpytorch.kernels.RBFKernel()
        k3 = gpytorch.kernels.RBFKernel()
        sig = torch.sigmoid(real_data[:, :2])
        equivalent = k1(sig) + k2(sig) + k3(sig)
        K_eq = equivalent.evaluate() / 3

        indices = np.random.choice(np.array(real_n), size=10)
        for i in indices:
            self.assertAlmostEqual(K_proj[i, 0].detach().numpy(), K_eq[i, 0].detach().numpy(), places=5)

    def test_weighted(self):
        kernel = ProjectionKernel(self.J, self.k, self.d, self.base_kernels, self.Ws, self.bs, weighted=True)
        param_dict = dict()
        for name, param in kernel.named_parameters():
            param_dict[name] = param
        self.assertEqual(len(param_dict), self.J*2)
        self.assertEqual(param_dict['base_kernels.1.base_kernel.raw_lengthscale'], 0)
        self.assertEqual(param_dict['base_kernels.1.raw_outputscale'], 0)

    def test_learn_proj(self):
        kernel = ProjectionKernel(self.J, self.k, self.d, self.base_kernels, self.Ws, self.bs, learn_proj=True)
        self.assertIsNone(kernel.last_x1)
        self.assertIsNone(kernel.cached_projections)
        _ = kernel(real_data, real_data).evaluate_kernel()
        self.assertIsNone(kernel.last_x1)
        self.assertIsNone(kernel.cached_projections)

        param_dict = dict()
        for name, param in kernel.named_parameters():
            param_dict[name] = param
        self.assertEqual(len(param_dict), self.J*(3))  # lengthscale, W, and b.

    pass


class TestCreateKernel(TestCase):
    pass
