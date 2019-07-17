from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors import GammaPrior
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gp_models import StrictlyAdditiveKernel, ExactGPModel, RPPolyKernel, ProjectionKernel
from fitting.optimizing import train_to_convergence, mean_squared_error, learn_projections
from training_routines import create_strictly_additive_kernel, create_additive_rp_kernel
import torch
import gpytorch
from math import sqrt, pi
import numpy as np
import matplotlib.pyplot as plt
import gc
import json
import gpytorch.settings as gp_set
from gp_experiment_runner import run_experiment

device = 'cuda:0'


########## FUNCTIONS ####################
def unimodal_d_dim(x):
    n, d = x.shape
    return torch.exp(- torch.norm(x, dim=1)**2)

def bimodal_d_dim(x):
    n, d = x.shape
    return torch.exp(- torch.norm(x + torch.ones(1, d).to(x), dim=1)) + \
           torch.exp(- torch.norm(x - torch.ones(1, d).to(x), dim=1))

def multimodal_d_dim(x):
    n, d = x.shape
    centers = [torch.zeros(1, d).to(x) for _ in range(d)]
    for i in range(d):
        centers[i][0,i] = 1.
        centers[i][0,:i] = -1.
        centers[i][0,i+1:] = -1.
    total = 0
    for i in range(d):
        total = total + torch.exp(-torch.norm(x - centers[i], dim=1))
    return total

def leading_dim(x):
    n, d = x.shape
    l = torch.sin(x[:, 0] * pi) * 1.
    return bimodal_d_dim(x[:, 1:])*0.4 + l

def one_dim(x):
    n, d = x.shape
    l = torch.sin(x[:, 0] * pi) * 1.
    return l

def half_relevant(x):
    n, d = x.shape
    dprime = d//2
    return unimodal_d_dim(x[:, :dprime])

def nonseparable(x):
    n, d = x.shape
    return x.prod(dim=-1)


def non_additive(x):
    n, d = x.shape
    # Continuous XOR by mixture of Gaussians
    centers = torch.eye(d)*1.4
    centers = torch.cat([centers, -centers]).t()
    centers = centers.repeat(n, 1, 1)
    y = torch.exp(-3 * (x.unsqueeze(2).expand_as(centers) - centers).pow(2).sum(dim=1)).sum(dim=1)
    return y


def benchmark_on_n_pts(n_pts, create_model_func, target_func, ho_x, ho_y, fit=True, repeats=3, max_iter=1000, return_model=False, verbose=0, checkpoint=True, print_freq=1, **kwargs):
    dims = ho_x.shape[1]
#     if n_pts > 20:
#         ho_x = ho_x.to(device)
#         ho_y = ho_y.to(device)
    rep_mses = []
    models = []
    mlls = []
    for i in range(repeats):
        # Don't edit the master copies fo the hold-out dataset 
        test_ho_x = torch.empty_like(ho_x).copy_(ho_x)
        test_ho_y = torch.empty_like(ho_y).copy_(ho_y)
        
#         test_ho_x = ho_x.copy_()
#         test_ho_y = ho_y.copy_()
    
        
        # Create the data.
        data = torch.rand(n_pts, dims)*4 - 2
        y = target_func(data) + torch.randn(n_pts)*0.01
        
        # Normalize by TEST in this case for all methods for more accurate comparison
        m = ho_x.mean(dim=0)
        s = ho_x.std(dim=0)
        data = (data - m) / s
        test_ho_x = (test_ho_x - m) / s
        # Do the same for Ys.
        m = ho_y.mean()
        s = ho_y.std()
        y = (y - m) / s
        test_ho_y = (test_ho_y - m) / s
        
        
        # Create the model now.
        model = create_model_func(data, y, **kwargs)

        # Put things on the GPU if necessary
        if n_pts > 20:
            test_ho_x = test_ho_x.to(device)
            test_ho_y = test_ho_y.to(device)
            model = model.to(device)
            data = data.to(device)
            y = y.to(device)
        with gp_set.fast_computations(True, True, True), gp_set.max_cg_iterations(10_000):
            with gp_set.cg_tolerance(0.001), gp_set.eval_cg_tolerance(0.0005), gp_set.memory_efficient(True):
                if fit:
                    mll = ExactMarginalLogLikelihood(model.likelihood, model)
                    train_to_convergence(model, data, y, torch.optim.Adam, objective=mll, checkpoint=checkpoint, 
                                         max_iter=max_iter, print_freq=print_freq, verbose=verbose)
                model.eval()
                with torch.no_grad():
                    mse = mean_squared_error(model(test_ho_x).mean, test_ho_y)
                    print(i, mse)
                    rep_mses.append(mse)
                    if return_model:
                        models.append(model)
                        mlls.append(mll)
                    else:
                        del mll
                        del model
                    del data
                    del y
    del ho_x
    del ho_y        
        
    torch.cuda.empty_cache()
    gc.collect()
    return rep_mses, models, mlls


def benchmark_algo_on_func(create_model_func, target_func, dims=6, max_pts=2560, fit=True, repeats=3, start_after=0, **kwargs):
    identifier = np.random.randint(0, 1e9)
    file = './progress_log_{:09d}.json'.format(identifier)
    print(file)
    rmses = []
    ho_x = torch.rand(4000, dims)*4 - 2
    ho_y = target_func(ho_x) 
    for n_pts in (10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240):
        if n_pts <= start_after:
            continue
        if n_pts > max_pts:
            break
        print('n_pts={}'.format(n_pts))
        rep_mses, _, _ = benchmark_on_n_pts(n_pts, create_model_func, target_func, ho_x, ho_y, fit=fit, repeats=repeats, **kwargs)
        rmses.append(np.mean(np.sqrt(rep_mses)))
        json.dump(rmses, open(file, 'w'))
    return rmses


################## MODELS ###########################
def create_bl_model(data, y):
    kernel = ScaleKernel(MaternKernel())
    model = ExactGPModel(data, y, GaussianLikelihood(), kernel)
    return model


def create_rp_model(data, y, proj_ratio=1):
    n, d = data.shape
    kernel = ScaleKernel(RPPolyKernel(round(proj_ratio * d), 1, d, MaternKernel, nu=2.5, weighted=True, 
                                      space_proj=True))
    model = ExactGPModel(data, y, GaussianLikelihood(), kernel)
    return model


def create_poly_rp_model(data, y, J, k):
    n, d = data.shape
    kernel = ScaleKernel(RPPolyKernel(J, k, d, RBFKernel, weighted=True, 
                                      space_proj=True))
    model = ExactGPModel(data, y, GaussianLikelihood(), kernel)
    return model


def create_dpa_gp_ard_model(data, y, J):
    n, d = data.shape
    kernel = ScaleKernel(create_additive_rp_kernel(d, J, learn_proj=False, kernel_type='RBF', space_proj=True, prescale=True, batch_kernel=False, ard=True, proj_dist='sphere', mem_efficient=True))
    model = ExactGPModel(data, y, GaussianLikelihood(), kernel)
    return model


def create_gam_model(data, y):
    n, d = data.shape
    kernel = ScaleKernel(create_strictly_additive_kernel(d, False, 'RBF', memory_efficient=True))
    model = ExactGPModel(data, y, GaussianLikelihood(), kernel)
    return model






############# Configs #################

dims = 2
max_pts = 20_000
repeats = 30
func = non_additive
output_fname = 'test_synth_experiment_2d.json'

rbf_rmses = benchmark_algo_on_func(create_bl_model, func, dims=dims, max_pts=max_pts, repeats=repeats)
gam_rmses = benchmark_algo_on_func(create_gam_model, func, dims=dims, max_pts=max_pts, repeats=repeats)
dpa_rmses = benchmark_algo_on_func(create_rp_model, func, dims=dims, max_pts=max_pts, repeats=repeats)
dpa_ard_rmses = benchmark_algo_on_func(create_dpa_gp_ard_model, func, dims=dims, max_pts=max_pts, repeats=repeats, J=dims)

json.dump({'rbf': rbf_rmses, 'gam': gam_rmses, 'dpa': dpa_rmses, 'dpa_ard': dpa_ard_rmses}, open('./run_outputs/{}'.format(output_fname), 'w'))

