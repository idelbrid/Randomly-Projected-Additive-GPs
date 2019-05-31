from unittest import TestCase
from gp_models import ProjectionKernel, PolynomialProjectionKernel, ExactGPModel, GeneralizedPolynomialProjectionKernel
from gp_models import AdditiveKernel, StrictlyAdditiveKernel, convert_rp_model_to_additive_model, DuvenaudAdditiveKernel
from gp_models import ManualRescaleProjectionKernel, MemoryEfficientGamKernel
from gp_models.models import AdditiveExactGPModel, ProjectedAdditiveExactGPModel
from rp import gen_rp, space_equally
from gp_experiment_runner import load_dataset, _normalize_by_train, _access_fold, _determine_folds
import torch
import torch.nn.functional as F
import gpytorch
from gpytorch.kernels import RBFKernel, ScaleKernel, AdditiveStructureKernel
import numpy as np
import pandas as pd


fake_data = 10+torch.randn(100, 100)*30   # 100 features, 100 points
fake_target = torch.sin(fake_data[:, 0]) + torch.cos(fake_data[:, 1])

more_fake_data = 10+torch.randn(100, 2)*30   # 2 features, 100 points
more_fake_target = torch.sin(fake_data[:, 0]) + torch.cos(fake_data[:, 1])
even_more_fake_data = 10+torch.randn(20, 2)*30   # 2 features, 30 points

real_data = torch.Tensor(load_dataset('breastcancer').iloc[:, 1:-1].values)
real_n, real_d = real_data.shape
real_target = torch.Tensor(load_dataset('breastcancer').iloc[:, -1].values)


class TwoLayerNN(torch.nn.Module):
    """Note: linear output"""
    def __init__(self, input_dim, hidden_units, output_dim):
        super(TwoLayerNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_units)
        self.fc2 = torch.nn.Linear(hidden_units, output_dim)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x


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
        self.assertEqual(len(param_dict), self.J*(self.k + 1) + 2)  # lengthscales + mixins
        self.assertEqual(param_dict['kernel.kernels.0.base_kernel.kernels.0.raw_lengthscale'], 0)
        self.assertIn('kernel.kernels.0.base_kernel.kernels.1.raw_lengthscale', param_dict.keys())
        self.assertIn('kernel.kernels.2.base_kernel.kernels.0.raw_lengthscale', param_dict.keys())
        self.assertIn('kernel.kernels.2.raw_outputscale', param_dict.keys())
        self.assertIn('kernel.kernels.0.raw_outputscale', param_dict.keys())
        self.assertFalse(kernel.kernel.kernels[0].raw_outputscale.requires_grad)
        self.assertFalse(kernel.projection_module.weight.requires_grad)
        self.assertFalse(kernel.projection_module.bias.requires_grad)

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

    def test_weighted(self):
        kernel = PolynomialProjectionKernel(self.J, self.k, self.d, self.base_kernel, self.Ws, self.bs, activation=None, learn_proj=False, weighted=True)
        param_dict = dict()
        for name, param in kernel.named_parameters():
            param_dict[name] = param
        self.assertEqual(len(param_dict), self.J*(self.k + 1) + 2)  # lengthscales + mixins
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
        self.assertTrue(kernel.projection_module.bias.requires_grad)
        self.assertTrue(kernel.projection_module.weight.requires_grad)

    def test_fit(self):
        kernel = PolynomialProjectionKernel(self.J, self.k, self.d,
                                            self.base_kernel, self.Ws, self.bs,
                                            learn_proj=False,
                                            weighted=False)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(real_data, real_target, likelihood, kernel)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        self.assertAlmostEqual(kernel.kernel.kernels[0].outputscale.item(), 1/self.J)
        self.assertAlmostEqual(kernel.projection_module.weight[0, 0], 1)
        self.assertAlmostEqual(kernel.projection_module.weight[0, 1], 0)
        model.train()

        optimizer_ = torch.optim.Adam(model.parameters(), lr=0.01)
        optimizer_.zero_grad()
        output = model(real_data)
        loss = -mll(output, real_target)
        loss.backward()
        optimizer_.step()
        self.assertNotEqual(kernel.kernel.kernels[0].base_kernel.kernels[0].raw_lengthscale, 0)
        self.assertAlmostEqual(kernel.kernel.kernels[0].outputscale.item(), 1/self.J)
        self.assertAlmostEqual(kernel.projection_module.weight[0, 0], 1)
        self.assertAlmostEqual(kernel.projection_module.weight[0, 1], 0)

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
        self.assertAlmostEqual(kernel.projection_module.weight[0, 0], 1)
        self.assertAlmostEqual(kernel.projection_module.weight[0, 1], 0)
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
        self.assertAlmostEqual(kernel.projection_module.weight[0, 0], 1)
        self.assertAlmostEqual(kernel.projection_module.weight[0, 1], 0)

    def test_fit_learn_proj(self):
        kernel = PolynomialProjectionKernel(self.J, self.k, self.d,
                                            self.base_kernel, self.Ws, self.bs,
                                            activation=None, learn_proj=True,
                                            weighted=False)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(real_data, real_target, likelihood, kernel)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        self.assertAlmostEqual(kernel.kernel.kernels[0].outputscale.item(), 1/self.J)
        self.assertAlmostEqual(kernel.projection_module.weight[0, 0], 1)
        self.assertAlmostEqual(kernel.projection_module.weight[0, 1], 0)
        old_val = float(kernel.projection_module.weight[0,0].item())
        model.train()

        optimizer_ = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(10):
            optimizer_.zero_grad()
            output = model(real_data)
            loss = -mll(output, real_target)
            loss.backward()
            optimizer_.step()
        self.assertNotEqual(kernel.kernel.kernels[0].base_kernel.kernels[0].raw_lengthscale, 0)
        self.assertAlmostEqual(kernel.kernel.kernels[0].outputscale.item(), 1/self.J)
        self.assertNotEqual(kernel.projection_module.weight[0, 0].item(), old_val)
        self.assertNotAlmostEqual(kernel.projection_module.weight[0, 1].item(), 0)


class TestGeneralPolyProjKernel(TestCase):
    def setUp(self):
        self.k = 2
        self.d = real_d
        self.J = 3
        self.projection = TwoLayerNN(self.d, 10, self.J*self.k)
        self.base_kernel = gpytorch.kernels.RBFKernel

    def test_init(self):
        kernel = GeneralizedPolynomialProjectionKernel(
            self.J, self.k, self.d, self.base_kernel, self.projection,
            learn_proj=False, weighted=False)
        param_dict = dict()
        for name, param in kernel.named_parameters():
            param_dict[name] = param
        self.assertEqual(len(param_dict),
                         self.J * (self.k + 1) + 4)  # lengthscales + mixins + weight/bias + weight/bias
        self.assertEqual(param_dict[
                             'kernel.kernels.0.base_kernel.kernels.0.raw_lengthscale'],
                         0)
        self.assertIn('kernel.kernels.0.base_kernel.kernels.1.raw_lengthscale',
                      param_dict.keys())
        self.assertIn('kernel.kernels.2.base_kernel.kernels.0.raw_lengthscale',
                      param_dict.keys())
        self.assertIn('kernel.kernels.2.raw_outputscale', param_dict.keys())
        self.assertIn('kernel.kernels.0.raw_outputscale', param_dict.keys())
        self.assertFalse(kernel.kernel.kernels[0].raw_outputscale.requires_grad)
        self.assertFalse(kernel.projection_module.fc1.weight.requires_grad)
        self.assertFalse(kernel.projection_module.fc2.weight.requires_grad)
        self.assertFalse(kernel.projection_module.fc2.bias.requires_grad)

    def test_initialize(self):
        kernel = GeneralizedPolynomialProjectionKernel(
            self.J, self.k, self.d, self.base_kernel, self.projection,
            learn_proj=False, weighted=False)
        kernel.initialize([0.1, 0.1], [0.1, 0.1])

        self.assertAlmostEqual(kernel.kernel.kernels[0].outputscale.item(), 1/self.J)
        self.assertAlmostEqual(kernel.kernel.kernels[0].base_kernel.kernels[0].lengthscale.item(), 0.1)
        self.assertFalse(kernel.kernel.kernels[0].raw_outputscale.requires_grad)

        kernel = GeneralizedPolynomialProjectionKernel(
            self.J, self.k, self.d, self.base_kernel, self.projection,
            learn_proj=False, weighted=True)
        kernel.initialize([0.1, 0.1], [0.1, 0.1])

        self.assertAlmostEqual(kernel.kernel.kernels[0].outputscale.item(), 1 / self.J)
        self.assertAlmostEqual(kernel.kernel.kernels[0].base_kernel.kernels[0].lengthscale.item(), 0.1)
        self.assertTrue(kernel.kernel.kernels[0].raw_outputscale.requires_grad)

    def test_ski(self):
        kernel = GeneralizedPolynomialProjectionKernel(
            self.J, self.k, self.d, self.base_kernel, self.projection,
            learn_proj=False, weighted=False, ski=True, ski_options=dict(grid_size=1000, num_dims=1))
        kernel.initialize([0.1, 0.2], [0.1, 0.2])
        kernel(real_data)
        self.assertIsInstance(kernel.kernel.kernels[0].base_kernel.kernels[0], gpytorch.kernels.GridInterpolationKernel)


class TestStrictlyAdditiveKernel(TestCase):
    def test_init(self):
        kernel = StrictlyAdditiveKernel(real_d, gpytorch.kernels.RBFKernel)
        self.assertIsInstance(kernel.kernel, gpytorch.kernels.AdditiveKernel)
        self.assertIsInstance(kernel.kernel.kernels[0].base_kernel.kernels[0], gpytorch.kernels.RBFKernel)

    def test_initialize(self):
        kernel = StrictlyAdditiveKernel(real_d, gpytorch.kernels.RBFKernel)
        kernel.initialize([0.1, 0.1], [0.1, 0.1])

    def test_forward(self):
        kernel = StrictlyAdditiveKernel(real_d, gpytorch.kernels.RBFKernel)
        kernel.forward(real_data, real_data)


class TestAdditiveKernel(TestCase):
    def test_init(self):
        kernel = AdditiveKernel([[1, 2], [0, 3]], 4, gpytorch.kernels.RBFKernel)
        self.assertIsInstance(kernel.kernel.kernels[0].base_kernel.kernels[0], gpytorch.kernels.RBFKernel)

    def test_forward(self):
        kernel = AdditiveKernel([[1, 2], [0, 3]], 4, gpytorch.kernels.RBFKernel)
        kernel2 = AdditiveKernel([[0, 1], [2, 3]], 4, gpytorch.kernels.RBFKernel)
        x = torch.tensor([[0, 1, 2, 3],
                          [0, 1, 4, 5]], dtype=torch.float)
        k = kernel(x).evaluate()
        k2 = kernel2(x).evaluate()
        self.assertNotAlmostEqual(k[0, 1].item(), k2[0,1].item())

    def test_convert_from_projection_kernel(self):
        Ws = [torch.eye(2, 2) for _ in range(3)]
        bs = [torch.zeros(2) for _ in range(3)]
        kernel = PolynomialProjectionKernel(3, 2, 2, RBFKernel, Ws, bs, learn_proj=False, weighted=True)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(more_fake_data, more_fake_target, likelihood, kernel)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        optimizer_ = torch.optim.Adam(model.parameters(), lr=0.01)
        optimizer_.zero_grad()
        output = model(more_fake_data)
        loss = -mll(output, more_fake_target)
        loss.backward()
        optimizer_.step()

        add_model, projection = convert_rp_model_to_additive_model(model, True)
        z = projection(even_more_fake_data)
        add_model.eval()
        predictions = add_model(z)
        model.eval()
        expected_predictions = model(even_more_fake_data)
        self.assertListEqual(predictions.mean.tolist(), expected_predictions.mean.tolist())



class TestAdditivePredictions(TestCase):
    def test_additive_kernel(self):
        kernel = StrictlyAdditiveKernel(2, RBFKernel)
        lik = gpytorch.likelihoods.GaussianLikelihood()
        trainX = torch.tensor([[0, 0]], dtype=torch.float)
        trainY = torch.tensor([2], dtype=torch.float)
        testXequi = torch.tensor([[1, 1]], dtype=torch.float)
        testXdiff = torch.tensor([[1, 0]], dtype=torch.float)

        kernel(trainX, testXdiff).evaluate()
        model = AdditiveExactGPModel(trainX, trainY, lik, kernel)

        model.eval()
        equi_pred = model.additive_pred(testXequi)
        diff_pred = model.additive_pred(testXdiff)

        self.assertEqual(equi_pred[0].mean[0], equi_pred[1].mean[0])
        self.assertNotEqual(diff_pred[0].mean[0], diff_pred[1].mean[0])

        combined = diff_pred[0] + diff_pred[1]
        total = model(testXdiff)
        self.assertEqual(combined.mean, total.mean)
        # self.assertEqual(combined.covariance_matrix, total.covariance_matrix)

        testXlarger = torch.tensor([[1, 0], [1.1, 1.2]], dtype=torch.float)
        pred_larger = model.additive_pred(testXlarger)

        trainX = torch.rand(200, 10, dtype=torch.float)
        trainY = torch.sin(trainX[:, 0]) + torch.sin(trainX[:, 1]) + torch.randn(200, dtype=torch.float) * 0.5
        kernel = StrictlyAdditiveKernel(2, RBFKernel)
        lik = gpytorch.likelihoods.GaussianLikelihood().to(dtype=torch.float)
        kernel = kernel.to(dtype=torch.float)
        model = AdditiveExactGPModel(trainX, trainY, lik, kernel)
        testXmuchlarger = torch.rand(2000, 10, dtype=torch.float)
        preds = model.additive_pred(testXmuchlarger)
        for p in preds:
            p.sample(torch.Size([1]))

    def test_scale_kernel(self):
        kernel = ScaleKernel(StrictlyAdditiveKernel(2, RBFKernel))
        lik = gpytorch.likelihoods.GaussianLikelihood()
        trainX = torch.tensor([[0, 0]], dtype=torch.float)
        trainY = torch.tensor([2], dtype=torch.float)
        testXequi = torch.tensor([[1, 1]], dtype=torch.float)
        testXdiff = torch.tensor([[1, 0]], dtype=torch.float)

        kernel(trainX, testXdiff).evaluate()
        model = AdditiveExactGPModel(trainX, trainY, lik, kernel)

        model.eval()
        equi_pred = model.additive_pred(testXequi)
        diff_pred = model.additive_pred(testXdiff)

        self.assertEqual(equi_pred[0].mean[0], equi_pred[1].mean[0])
        self.assertNotEqual(diff_pred[0].mean[0], diff_pred[1].mean[0])

        combined = diff_pred[0] + diff_pred[1]
        total = model(testXdiff)
        self.assertEqual(combined.mean, total.mean)
        # self.assertEqual(combined.covariance_matrix, total.covariance_matrix)

        testXlarger = torch.tensor([[1, 0], [1.1, 1.2]], dtype=torch.float)
        pred_larger = model.additive_pred(testXlarger)

        trainX = torch.rand(200, 10, dtype=torch.float)
        trainY = torch.sin(trainX[:, 0]) + torch.sin(trainX[:, 1]) + torch.randn(200, dtype=torch.float) * 0.5
        kernel = ScaleKernel(StrictlyAdditiveKernel(2, RBFKernel))
        lik = gpytorch.likelihoods.GaussianLikelihood().to(dtype=torch.float)
        kernel = kernel.to(dtype=torch.float)
        model = AdditiveExactGPModel(trainX, trainY, lik, kernel)
        testXmuchlarger = torch.rand(2000, 10, dtype=torch.float)
        preds = model.additive_pred(testXmuchlarger)
        for p in preds:
            p.sample(torch.Size([1]))


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


class TestDuvenaudKernel(TestCase):
    def test_degree1(self):
        AddK = DuvenaudAdditiveKernel(3, 1)
        self.assertEqual(AddK.lengthscale.numel(), 3)
        self.assertEqual(AddK.outputscale.numel(), 1)

        testvals = torch.tensor([[1, 2, 3], [7, 5, 2]], dtype=torch.float)
        add_k_val = AddK(testvals, testvals).evaluate()

        manual_k = ScaleKernel(gpytorch.kernels.AdditiveKernel(RBFKernel(active_dims=0),
                                                               RBFKernel(active_dims=1),
                                                               RBFKernel(active_dims=2)))
        manual_add_k_val = manual_k(testvals, testvals).evaluate()

        self.assertTrue(torch.allclose(add_k_val, manual_add_k_val))

    def test_degree2(self):
        AddK = DuvenaudAdditiveKernel(3, 2)
        self.assertEqual(AddK.lengthscale.numel(), 3)
        self.assertEqual(AddK.outputscale.numel(), 2)

        testvals = torch.tensor([[1, 2, 3], [7, 5, 2]], dtype=torch.float)
        add_k_val = AddK(testvals, testvals).evaluate()

        manual_k1 = ScaleKernel(gpytorch.kernels.AdditiveKernel(RBFKernel(active_dims=0),
                                                               RBFKernel(active_dims=1),
                                                               RBFKernel(active_dims=2)))
        manual_k1.initialize(outputscale=1/3)
        manual_k2 = ScaleKernel(gpytorch.kernels.AdditiveKernel(RBFKernel(active_dims=[0,1]),
                                                                RBFKernel(active_dims=[1,2]),
                                                                RBFKernel(active_dims=[0,2])))
        manual_k2.initialize(outputscale=1/3)
        manual_k = gpytorch.kernels.AdditiveKernel(manual_k1, manual_k2)
        manual_add_k_val = manual_k(testvals, testvals).evaluate()

        self.assertTrue(torch.allclose(add_k_val, manual_add_k_val))

    def test_degree3(self):
        AddK = DuvenaudAdditiveKernel(2, 3)
        self.assertEqual(AddK.lengthscale.numel(), 2)
        self.assertEqual(AddK.outputscale.numel(), 2)

        testvals = torch.tensor([[2, 3], [5, 2]], dtype=torch.float)
        add_k_val = AddK(testvals, testvals).evaluate()


class TestManualRescaleKernel(TestCase):
    def test_prescale(self):
        x = torch.tensor([[1., 2., 3.], [1.1, 2.2, 3.3]])
        kbase = RBFKernel()
        kbase.initialize(lengthscale=torch.tensor([1.]))
        base_kernel = AdditiveStructureKernel(kbase, 3)
        proj_module = torch.nn.Linear(3, 3, bias=False)
        proj_module.weight.data = torch.eye(3, dtype=torch.float)
        proj_kernel = ManualRescaleProjectionKernel(proj_module, base_kernel, prescale=True, ard_num_dims=3)
        proj_kernel.initialize(lengthscale=torch.tensor([1., 2., 3.]))

        with torch.no_grad():
            K = proj_kernel(x, x).evaluate()

        k = RBFKernel()
        k.initialize(lengthscale=torch.tensor([1.]))

        with torch.no_grad():
            K2 = 3*k(x[:, 0:1], x[:, 0:1]).evaluate()

        np.testing.assert_allclose(K.numpy(), K2.numpy())

    def test_postscale(self):
        x = torch.tensor([[1., 2., 3.], [1.1, 2.2, 3.3]])
        kbase = RBFKernel()
        kbase.initialize(lengthscale=torch.tensor([1.]))
        base_kernel = AdditiveStructureKernel(kbase, 3)
        proj_module = torch.nn.Linear(3, 3, bias=False)
        proj_module.weight.data = torch.eye(3, dtype=torch.float)
        proj_kernel = ManualRescaleProjectionKernel(proj_module, base_kernel, prescale=False, ard_num_dims=3)
        proj_kernel.initialize(lengthscale=torch.tensor([1., 2., 3.]))

        with torch.no_grad():
            K = proj_kernel(x, x).evaluate()

        k = RBFKernel()
        k.initialize(lengthscale=torch.tensor([1.]))

        with torch.no_grad():
            K2 = 3 * k(x[:, 0:1], x[:, 0:1]).evaluate()

        np.testing.assert_allclose(K.numpy(), K2.numpy())

    def test_gradients(self):
        x = torch.tensor([[1., 2., 3.], [1.1, 2.2, 3.3]])
        y = torch.sin(x).sum(dim=1)
        kbase = RBFKernel()
        kbase.initialize(lengthscale=torch.tensor([1.]))
        base_kernel = AdditiveStructureKernel(kbase, 3)
        proj_module = torch.nn.Linear(3, 3, bias=False)
        proj_module.weight.data = torch.eye(3, dtype=torch.float)
        proj_kernel = ManualRescaleProjectionKernel(proj_module, base_kernel, prescale=True, ard_num_dims=3)
        proj_kernel.initialize(lengthscale=torch.tensor([1., 2., 3.]))

        model = ExactGPModel(x, y, gpytorch.likelihoods.GaussianLikelihood(), proj_kernel)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        optimizer_ = torch.optim.Adam(model.parameters(), lr=0.1)
        optimizer_.zero_grad()

        pred = model(x)
        loss = -mll(pred, y)
        loss.backward()

        optimizer_.step()

        np.testing.assert_allclose(proj_kernel.base_kernel.base_kernel.lengthscale.numpy(), torch.tensor([[1.]]).numpy())
        np.testing.assert_allclose(proj_kernel.projection_module.weight.numpy(), torch.eye(3, dtype=torch.float).numpy())
        self.assertFalse(np.allclose(proj_kernel.lengthscale.detach().numpy(), torch.tensor([1., 2., 3.]).numpy()))

        proj_module = torch.nn.Linear(3, 3, bias=False)
        proj_module.weight.data = torch.eye(3, dtype=torch.float)
        proj_kernel2 = ManualRescaleProjectionKernel(proj_module, base_kernel, prescale=True, ard_num_dims=3,
                                                     learn_proj=True)

        proj_kernel2.initialize(lengthscale=torch.tensor([1., 2., 3.]))

        model = ExactGPModel(x, y, gpytorch.likelihoods.GaussianLikelihood(), proj_kernel2)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        optimizer_ = torch.optim.Adam(model.parameters(), lr=0.1)
        optimizer_.zero_grad()

        pred = model(x)
        loss = -mll(pred, y)
        loss.backward()

        optimizer_.step()

        np.testing.assert_allclose(proj_kernel2.base_kernel.base_kernel.lengthscale.numpy(), torch.tensor([[1.]]).numpy())
        self.assertFalse(np.allclose(proj_kernel2.projection_module.weight.detach().numpy(), torch.eye(3, dtype=torch.float).numpy()))
        self.assertFalse(np.allclose(proj_kernel2.lengthscale.detach().numpy(), torch.tensor([1., 2., 3.]).numpy()))


class TestMemEffGamKernel(TestCase):
    def test_forward(self):
        gam_kernel = MemoryEfficientGamKernel()
        x = torch.tensor([[1., 2., 3.], [1.1, 2.2, 3.3]])
        K = gam_kernel(x, x).evaluate()

        k = ScaleKernel(RBFKernel())
        k.initialize(outputscale=1/3)
        as_kernel = AdditiveStructureKernel(k, 2)
        K2 = as_kernel(x, x).evaluate()

        np.testing.assert_allclose(K.detach().numpy(), K2.detach().numpy(), atol=1e-6)