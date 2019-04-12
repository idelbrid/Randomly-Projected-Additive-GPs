from unittest import TestCase
from gp_helpers import ProjectionKernel, PolynomialProjectionKernel, ExactGPModel
from rp import gen_rp, space_equally
from rp_experiments import load_dataset, _normalize_by_train, _access_fold, _determine_folds
import torch
import gpytorch
import numpy as np
import pandas as pd


fake_data = 10+torch.randn(100, 100)*30   # 100 features, 100 points
fake_target = torch.sin(fake_data[:, 0]) + torch.cos(fake_data[:, 1])

real_data = torch.Tensor(load_dataset('breastcancer').iloc[:, 1:-1].values)
real_n, real_d = real_data.shape
real_target = torch.Tensor(load_dataset('breastcancer').iloc[:, -1].values)


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

        self.assertAlmostEqual(W[:, 0].norm().item(), W[:, 1].norm().item())

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


class TestPolyProjectionKernel(TestCase):
    def setUp(self):
        self.k = 2
        self.d = real_d
        self.J = 3
        self.Ws = [torch.eye(real_d, 2) for _ in range(self.J)]
        self.bs = [torch.zeros(2) for _ in range(self.J)]
        self.base_kernel = gpytorch.kernels.RBFKernel

    def test_init(self):
        kernel = PolynomialProjectionKernel(self.J, self.k, self.d, self.base_kernel, self.Ws, self.bs, activation=None, learn_proj=False, weighted=False)
        param_dict = dict()
        for name, param in kernel.named_parameters():
            param_dict[name] = param
        self.assertEqual(len(param_dict), self.J*(self.k + 1))  # lengthscales + mixins
        self.assertEqual(param_dict['kernel.kernels.0.base_kernel.kernels.0.raw_lengthscale'], 0)
        self.assertIn('kernel.kernels.0.base_kernel.kernels.1.raw_lengthscale', param_dict.keys())
        self.assertIn('kernel.kernels.2.base_kernel.kernels.0.raw_lengthscale', param_dict.keys())
        self.assertIn('kernel.kernels.2.raw_outputscale', param_dict.keys())
        self.assertIn('kernel.kernels.0.raw_outputscale', param_dict.keys())
        self.assertFalse(kernel.kernel.kernels[0].raw_outputscale.requires_grad)
        self.assertNotIn('W', param_dict.keys())
        self.assertNotIn('b', param_dict.keys())

    def test_forward(self):
        kernel = PolynomialProjectionKernel(self.J, self.k, self.d, self.base_kernel, self.Ws, self.bs, activation=None, learn_proj=False, weighted=False)
        output = kernel(real_data, real_data)
        self.assertIsInstance(output, gpytorch.lazy.LazyEvaluatedKernelTensor)
        evl = output.evaluate_kernel()
        self.assertIsInstance(evl, gpytorch.lazy.SumLazyTensor)
        # TODO: how to validate that we're calculating using SKIP?

        K_proj = output.evaluate()

        k1 = gpytorch.kernels.RBFKernel()
        k2 = gpytorch.kernels.RBFKernel()
        k3 = gpytorch.kernels.RBFKernel()
        equivalent = k1(real_data[:, :1])*k1(real_data[:, 1:2]) + k2(real_data[:, :1])*k2(real_data[:, 1:2]) + k3(real_data[:, :1])*k3(real_data[:, 1:2])
        K_eq = equivalent.evaluate() / 3

        indices = np.random.choice(np.array(real_n), size=10)
        for i in indices:
            self.assertAlmostEqual(K_proj[i, 0].detach().numpy(), K_eq[i, 0].detach().numpy(), places=5)

    def test_caching(self):
        kernel = PolynomialProjectionKernel(self.J, self.k, self.d, self.base_kernel, self.Ws, self.bs, activation=None, learn_proj=False, weighted=False)
        self.assertIsNone(kernel.last_x1)
        self.assertIsNone(kernel.cached_projections)
        _ = kernel(real_data, real_data).evaluate_kernel()
        self.assertIsNotNone(kernel.last_x1)
        self.assertIsNotNone(kernel.cached_projections)
        new = kernel(real_data[:10], real_data[:10]).evaluate()
        self.assertEqual(kernel.last_x1.numel(), real_data[:10].numel())
        self.assertEqual(new.numel(), 100)

    def test_activation(self):
        kernel = PolynomialProjectionKernel(self.J, self.k, self.d, self.base_kernel, self.Ws, self.bs, activation=None, learn_proj=False, weighted=False)
        output = kernel(real_data, real_data)
        K_proj = output.evaluate()

        k1 = gpytorch.kernels.RBFKernel()
        k2 = gpytorch.kernels.RBFKernel()
        k3 = gpytorch.kernels.RBFKernel()
        sig = torch.sigmoid(real_data[:, :2])
        equivalent = k1(sig[:, 0:1])*k1(sig[:, 1:2]) + k2(sig[:, 0:1])*k2(sig[:, 1:2]) + k3(sig[:, 0:1])*k3(sig[:, 1:2])
        K_eq = equivalent.evaluate() / 3

        indices = np.random.choice(np.array(real_n), size=10)
        for i in indices:
            self.assertAlmostEqual(K_proj[i, 0].detach().numpy(), K_eq[i, 0].detach().numpy(), places=5)

    def test_weighted(self):
        kernel = PolynomialProjectionKernel(self.J, self.k, self.d, self.base_kernel, self.Ws, self.bs, activation=None, learn_proj=False, weighted=True)
        param_dict = dict()
        for name, param in kernel.named_parameters():
            param_dict[name] = param
        self.assertEqual(len(param_dict), self.J*(self.k + 1))  # lengthscales + mixins
        self.assertIn('kernel.kernels.2.raw_outputscale', param_dict.keys())
        self.assertIn('kernel.kernels.0.raw_outputscale', param_dict.keys())
        self.assertTrue(kernel.kernel.kernels[0].raw_outputscale.requires_grad)

    def test_learn_proj(self):
        kernel = PolynomialProjectionKernel(self.J, self.k, self.d, self.base_kernel, self.Ws, self.bs, activation=None, learn_proj=True, weighted=False)
        self.assertIsNone(kernel.last_x1)
        self.assertIsNone(kernel.cached_projections)
        _ = kernel(real_data, real_data).evaluate_kernel()
        self.assertIsNone(kernel.last_x1)
        self.assertIsNone(kernel.cached_projections)

        param_dict = dict()
        for name, param in kernel.named_parameters():
            param_dict[name] = param
        self.assertEqual(len(param_dict), self.J*(self.k+1) + 2)  # outputscale, lengthscales for each j, plus W, and b.
        self.assertIn('W', param_dict.keys())
        self.assertIn('b', param_dict.keys())

    def test_fit(self):
        kernel = PolynomialProjectionKernel(self.J, self.k, self.d,
                                            self.base_kernel, self.Ws, self.bs,
                                            activation=None, learn_proj=False,
                                            weighted=False)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(real_data, real_target, likelihood, kernel)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        self.assertAlmostEqual(kernel.kernel.kernels[0].outputscale.item(), 1/self.J)
        self.assertAlmostEqual(kernel.W[0, 0], 1)
        self.assertAlmostEqual(kernel.W[0, 1], 0)
        model.train()

        optimizer_ = torch.optim.Adam(model.parameters(), lr=0.01)
        optimizer_.zero_grad()
        output = model(real_data)
        loss = -mll(output, real_target)
        loss.backward()
        optimizer_.step()
        self.assertNotEqual(kernel.kernel.kernels[0].base_kernel.kernels[0].raw_lengthscale, 0)
        self.assertAlmostEqual(kernel.kernel.kernels[0].outputscale.item(), 1/self.J)
        self.assertAlmostEqual(kernel.W[0, 0], 1)
        self.assertAlmostEqual(kernel.W[0, 1], 0)


    def test_fit_weighted(self):
        kernel = PolynomialProjectionKernel(self.J, self.k, self.d,
                                            self.base_kernel, self.Ws, self.bs,
                                            activation=None, learn_proj=False,
                                            weighted=True)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(real_data, real_target, likelihood, kernel)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        self.assertAlmostEqual(kernel.kernel.kernels[0].outputscale.item(),
                               1 / self.J)
        old_value = float(kernel.kernel.kernels[0].outputscale.item())
        self.assertAlmostEqual(kernel.W[0, 0], 1)
        self.assertAlmostEqual(kernel.W[0, 1], 0)
        model.train()

        optimizer_ = torch.optim.Adam(model.parameters(), lr=0.01)
        optimizer_.zero_grad()
        output = model(real_data)
        loss = -mll(output, real_target)
        loss.backward()
        optimizer_.step()
        self.assertNotEqual(
            kernel.kernel.kernels[0].base_kernel.kernels[0].raw_lengthscale, 0)
        self.assertNotEqual(kernel.kernel.kernels[0].outputscale.item(),
                            old_value)
        self.assertAlmostEqual(kernel.W[0, 0], 1)
        self.assertAlmostEqual(kernel.W[0, 1], 0)

    def test_fit_learn_proj(self):
        kernel = PolynomialProjectionKernel(self.J, self.k, self.d,
                                            self.base_kernel, self.Ws, self.bs,
                                            activation=None, learn_proj=True,
                                            weighted=False)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(real_data, real_target, likelihood, kernel)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        self.assertAlmostEqual(kernel.kernel.kernels[0].outputscale.item(), 1/self.J)
        self.assertAlmostEqual(kernel.W[0, 0], 1)
        self.assertAlmostEqual(kernel.W[0, 1], 0)
        old_val = float(kernel.W[0,0].item())
        model.train()

        optimizer_ = torch.optim.Adam(model.parameters(), lr=0.01)
        optimizer_.zero_grad()
        output = model(real_data)
        loss = -mll(output, real_target)
        loss.backward()
        optimizer_.step()
        self.assertNotEqual(kernel.kernel.kernels[0].base_kernel.kernels[0].raw_lengthscale, 0)
        self.assertAlmostEqual(kernel.kernel.kernels[0].outputscale.item(), 1/self.J)
        self.assertNotEqual(kernel.W[0, 0].item(), old_val)
        self.assertNotAlmostEqual(kernel.W[0, 1].item(), 0)


class TestSpaceEqually(TestCase):
    def test_d2J2(self):
        d = 2
        k = 1
        J = 2
        P = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        newP, _ = space_equally(P, 1.0, 10)
        self.assertListEqual(newP.numpy().tolist(), P.numpy().tolist())

        P = torch.tensor([[1/np.sqrt(2), 1/np.sqrt(2)], [1.0, 0.0]])
        newP, loss = space_equally(P, 0.1, 10000)
        self.assertAlmostEqual(loss.item(), 0)

    def testd4J2(self):
        d = 4
        k = 1
        J = 2
        P = torch.cat([gen_rp(d, k, dist='gaussian') for _ in range(J)],
                      dim=1).t()
        newP, loss = space_equally(P, 0.1, 10000)
        self.assertAlmostEqual(loss.item(), 0)

    def testd2J4(self):
        d = 2
        k = 1
        J = 4
        P = torch.cat([gen_rp(d, k, dist='gaussian') for _ in range(J)],
                      dim=1).t()
        newP, loss = space_equally(P, 0.1, 10000)
        self.assertNotAlmostEqual(loss.item(), 0)







class TestExperimentHelpers(TestCase):
    def testNormalizeByTrain(self):
        df = pd.DataFrame({'index': [0, 1, 2],
                           '0': [1, 2, 2],
                            'target': [0, 0, 1]})
        train = df.iloc[:2, :]
        test = df.iloc[2:, :]

        new_train, new_test = _normalize_by_train(train, test)
        mean = 0.5
        std = np.std([-0.5, 0.5], ddof=1)  # to match pandas.
        self.assertEqual(new_train['0'].iloc[0], (0 - mean)/ std)
        self.assertEqual(new_test['0'].iloc[0], (0 + mean) / std)
        self.assertEqual(new_test['target'].iloc[0], 1)
        self.assertEqual(new_train['target'].iloc[0], 0)

    def testDetermineFold(self):
        df = pd.DataFrame({'index': [0, 1, 2, 3],
                           '0': [1, 2, 2, 4],
                           'target': [0, 0, 1, 1]})
        fold_starts = _determine_folds(1/3, df)  # want 3 folds
        self.assertEqual(len(fold_starts), 4)
        self.assertListEqual(fold_starts, [0, 2, 3, 4])

    def testAccessFold(self):
        df = pd.DataFrame({'index': [0, 1, 2, 3],
                           '0': [1, 2, 2, 4],
                           'target': [0, 0, 1, 1]})
        fold_starts = [0, 2, 3, 4]
        train, test = _access_fold(df, fold_starts, 0)
        self.assertListEqual(test['index'].values.tolist(), [0,1])
        self.assertListEqual(train['index'].values.tolist(), [2, 3])

        train, test = _access_fold(df, fold_starts, 1)
        self.assertListEqual(test['index'].values.tolist(), [2])
        self.assertListEqual(train['index'].values.tolist(), [0, 1, 3])
