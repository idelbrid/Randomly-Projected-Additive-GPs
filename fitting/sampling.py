from utils import my_cdist
import gpytorch
import torch
from gp_models.models import ExactGPModel
import numpy as np

class MCMCSampler(object):
    def __init__(self):
        self.burn_in = None
        pass

    def sample(self, n_samples=1):
        pass

    def _sample(self):
        pass

    def predict(self, x):
        pass

class HMSampler(MCMCSampler):
    def __init__(self):
        super(HMSampler, self).__init__()

    def _propose(self):
        pass

    def _accept(self):
        pass

    def _sample(self):
        self._propose()

class GardnerMHSampler(HMSampler):
    def __init__(self):
        super(GardnerMHSampler, self).__init__()

    def _propose(self, ):
        pass
            # split = np.random.rand() < 0.5
            # if split:
            #     notsplit = True
            #     while notsplit:
            #         idx = np.random.randint(0, len(degrees))
            #         if degrees[idx] > 1:
            #             notsplit = False
            #             proposed = []
            #             for i, deg in enumerate(degrees):
            #                 if i == idx:
            #                     first = np.random.randint(1, deg)
            #                     proposed.append(first)
            #                     proposed.append(deg - first)
            #                 else:
            #                     proposed.append(deg)
            #     prob_forward = 0.5 * (1 / len(degrees)) * 2 ** (-degrees[idx] + 1)  # adjust!!!
            #     prob_backward = 0.5 * (1 / len(proposed)) * (1 / (len(proposed) - 1))
            # else:
            #     idx1, idx2 = np.random.choice(np.arange(len(degrees)), size=2, replace=False)
            #     proposed = [degrees[idx1] + degrees[idx2]]
            #     for i, deg in enumerate(degrees):
            #         if i != idx1 and i != idx2:
            #             proposed.append(deg)
            #
            #     prob_forward = 0.5 * (1 / len(degrees)) * (1 / (len(degrees) - 1))
            #     prob_backward = 0.5 * (1 / len(proposed)) * 2 ** (-proposed[0] + 1)  # adjust?
            # if split:
            #     print("SPLIT {} TO {}".format(degrees, proposed))
            # else:
            #     print("MERGE {} TO {}".format(degrees, proposed))
            # return proposed, prob_forward, prob_backward



class ModelAverage(object):
    def __init__(self, predictions, weights):
        self.predictions = predictions
        self.weights = weights  # logits
        norm_const = sum(np.exp(w) for w in weights)
        self.norm_weights = [np.exp(w)/norm_const for w in weights]

    def mean(self):  # TODO: Check that this is right...
        """Return the mean of the weighted sum of the densities
        This is just a weighted sum of means."""
        mean = torch.zeros_like(self.predictions[0].mean)
        for pred, w in zip(self.predictions, self.norm_weights):
            mean = mean + pred.mean * w

        return mean

    def log_prob(self, y):
        """Return the log probability represented by the densities of the individual densities"""
        prob = 0
        for pred, w in zip(self.predictions, self.norm_weights):
            prob = prob + torch.exp(pred.log_prob(y)) * w

        return torch.log(prob)

    def sample(self):
        idx = torch.distributions.Categorical(logits=torch.tensor(self.weights)).sample().item()
        return self.predictions[idx].sample()


class CGPSampler(object):
    def __init__(self, X, y, proj_dist='gaussian'):
        n, d = X.shape
        self.X = X
        self.y = y
        self.n = n
        self.d = d
        self.proj_dist = proj_dist
        self.num_ls_samples = 20
        self.ls_prior = gpytorch.priors.GammaPrior(1.6, 0.5)
        self.kernel = gpytorch.kernels.RBFKernel()
        self.min_k = np.ceil(2 * np.log(d))
        self.max_k = min(n, d)
        self.num_samples = self.max_k - self.min_k  # can be overwritten...
        self._dmax = None
        self._dmin = None

    @property
    def dmax(self):
        if self._dmax is None:
            self._get_dmax_dmin()
        return self._dmax

    @property
    def dmin(self):
        if self._dmin is None:
            self._get_dmax_dmin()
        return self._dmin

    def _get_dmax_dmin(self):
        d = my_cdist(self.X, self.X).pow_(2)
        self._dmax = d.max().item()
        self._dmin = d.min().item()

    def _sample_projection(self, k):
        """

        :param k:
        :return:
        """
        from rp import gen_rp
        return gen_rp(self.d, k, self.proj_dist)

    def _sample_lengthscale(self, projected_X):
        """

        :param projected_X:
        :return:
        """
        from numpy.random import choice
        # from scipy.special import gamma
        ls = []
        ws = []
        mls = []
        for i in range(self.num_ls_samples):
            l = torch.rand(1)*(self.dmax - self.dmin) + self.dmin  # sample lengthscale uniformly
            ls.append(l)

            # Compute marginal likelihood of data given lengthscale
            like = gpytorch.likelihoods.GaussianLikelihood()
            like.initialize(noise=1.)
            self.kernel.initialize(lengthscale=l)
            model = ExactGPModel(projected_X, self.y, like, self.kernel)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(like, model)
            mll.train()
            model.train()
            log_ml = mll(model(projected_X), self.y).item()
            # Posterior is basically proportional to prior * P(data | l)
            # consts = self.n/2 * np.log(2) + torch.lgamma(torch.tensor(self.n/2, dtype=torch.double)).to(torch.float)
            consts = 0  # don't actually need to compute the constants, since we are only ever taking a weighted avg
            w = (
                torch.tensor(log_ml) +
                self.ls_prior.log_prob(l) +
                consts
            )
            ws.append(w.item())
            mls.append(log_ml + consts)

        # wsum = sum(ws)
        # reweighted_ws = [w / wsum for w in ws]
        #
        # idx = choice(np.arange(self.num_ls_samples), p=reweighted_ws)
        cat = torch.distributions.Categorical(logits=torch.tensor(ws, dtype=torch.float))
        idx = cat.sample().item()
        return ls[idx], mls[idx]

    def pred(self, test_X):
        """Return an object wrapping the posterior distribution of the model average.

        :param test_X: n x d tensor of test data points.
        :return: ModelAverage object wrapping the
        """
        wghts = []
        preds = []
        for m in np.linspace(self.min_k, self.max_k, self.num_samples):
            m = int(np.floor(m))

            P = self._sample_projection(m)
            proj_X = self.X.matmul(P)
            lam, ml = self._sample_lengthscale(proj_X)
            self.kernel.initialize(lengthscale=lam)
            proj_test_X = test_X.matmul(P)
            like = gpytorch.likelihoods.GaussianLikelihood()
            like.initialize(noise=1.)
            self.kernel.initialize(lengthscale=lam)
            model = ExactGPModel(proj_X, self.y, like, self.kernel)
            model.eval()
            pred = model(proj_test_X)  # TODO: this is Gaussian, not t distributed.
            preds.append(pred)

            wghts.append(ml)  # TODO: Fix, the weights should be handled with log probabilities.

        return ModelAverage(preds, wghts)

