import bayesopt as bo
import torch
import gpytorch

import bayesopt.acquisition
import bayesopt.utils
import gp_models
from training_routines import create_rp_poly_kernel
import gp_experiment_runner
import training_routines
import matplotlib.pyplot as plt
import numpy as np
import scipy
from math import pi
from scipydirect import minimize
import pickle

def run_bo(d=10, iters=200, acq='ei', use_rp=False, repeats=5, learn_proj=False):
    histories = []
    for r in range(repeats):
        bounds = [(-4, 4) for _ in range(d)]
        if not use_rp:
            kernel = gpytorch.kernels.RBFKernel()
        else:
            kernel = create_rp_poly_kernel(d, 1, d, learn_proj=learn_proj, weighted=True, space_proj=True)

        kernel = gpytorch.kernels.ScaleKernel(kernel)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.initialize(noise=0.001)
        likelihood.raw_noise.requires_grad = False
        trainX, trainY = torch.tensor([]), torch.tensor([])
        model = gp_models.ExactGPModel(trainX, trainY, likelihood, kernel)
        training_options = dict(max_iter=10000, check_conv=True, lr=0.1, optimizer=torch.optim.Adam,
                                verbose=False, patience=20, smooth=True)
        if acq == 'ei':
            acq_fxn = bayesopt.acquisition.EI
        else:
            acq_fxn = bayesopt.acquisition.ThompsonSampling
        optimizer = bo.BayesOpt(bayesopt.utils.stybtang, bounds, gp_model=model, acq_fxn=acq_fxn, optimizer=minimize,
                                initial_points=15, init_method='latin_hc', gp_optim_freq=5,
                                gp_optim_options=training_options)

        with gpytorch.settings.fast_computations(False, False):
            optimizer.initialize()

        with gpytorch.settings.fast_computations(False, False):
            optimizer.steps(iters=iters-15, maxf=2000)
        histories.append(optimizer.true_best_y_path)

    return histories

hist = run_bo(10, 200, 'ei', False)
hist = np.array(hist)
np.save('my_implementation_ei_basic_kernel.npy', hist)

hist = run_bo(10, 200, 'Thompson', False)
hist = np.array(hist)
np.save('my_implementation_thompson_rp.npy', hist)

hist = run_bo(10, 200, 'ei', True)
hist = np.array(hist)
np.save('my_implementation_ei_rp.npy', hist)

hist = run_bo(10, 200, 'ei', True, learn_proj=True)
hist = np.array(hist)
np.save('my_implementation_ei_rp_learn_weights.npy', hist)