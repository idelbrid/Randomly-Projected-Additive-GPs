import argparse
from bayesopt.utils import stybtang, branin, michalewicz
from bayesopt import BayesOpt
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
from botorch.models import FixedNoiseGP, SingleTaskGP
from botorch.test_functions import hartmann6, eggholder
from gpytorch.kernels import ScaleKernel, MaternKernel, RBFKernel
from gpytorch.priors import GammaPrior
from gp_models import ExactGPModel, GeneralizedProjectionKernel, AdditiveKernel, \
    StrictlyAdditiveKernel
import json
import os
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--function', required=True, type=str)
    parser.add_argument('--kernel', required=True, type=str)
    parser.add_argument('--iters', default=200, type=int)
    parser.add_argument('--rebuild', action='store_true')
    parser.add_argument('--repeats')
    parser.add_argument('-f', '--file', required=True, type=str)
    parser.add_argument('-v', '--verbose', default=0, type=int)
    args = parser.parse_args()

    results = dict()
    results['call_options'] = args.__dict__

    train_yvar = torch.tensor([0.01], dtype=torch.float)
    def hook_factory(kernel_class, lengthscale_prior, outputscale_prior, **kwargs):
        def initialize_model_hook(trainX, trainY):
            model = FixedNoiseGP(trainX, trainY, train_yvar.expand_as(trainY)).to(trainX)
            model.covar_module = ScaleKernel(
                base_kernel=kernel_class(
                    ard_num_dims=trainX.shape[-1],
                    batch_shape=model._aug_batch_shape,
                    lengthscale_prior=lengthscale_prior,
                    **kwargs
                ),
                batch_shape=model._aug_batch_shape,
                outputscale_prior=outputscale_prior,
            )
            return model

        return initialize_model_hook

    # results['']
    for iteration in range(args.iters):
        iter_results = dict(iteration=iteration)
        try:
            optimizer = BayesOpt()
