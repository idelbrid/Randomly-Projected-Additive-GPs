import pandas as pd
import numpy as np
import torch
import gpytorch
from typing import Callable, Dict, Union
from gp_helpers import ExactGPModel, fit_gp_model, RPKernel, LinearRegressionModel, ELMModule, fit_linear_model
import time
import datetime
import rp
import traceback
import os
from scipy.io import loadmat
import json


# TODO: should I preprocess the data further by normalizing inputs/outputs?

def old_load_dataset(name: str):
    """Helper method to load a given dataset by name"""
    return pd.read_csv('./datasets/{}/dataset.csv'.format(name))


def load_dataset(name: str):
    mat = loadmat(os.path.join('.', 'uci', name, '{}.mat'.format(name)))
    [n, d] = mat['data'].shape
    df = pd.DataFrame(mat['data'],
                      columns=list(range(d-1))+['target'])
    df.columns = [str(c) for c in df.columns]
    df = df.reset_index()

    df['target'] = df['target'] - df['target'].mean()
    df['target'] = df['target']/(df['target'].std())

    # TODO: Figure out why this happens sometimes.
    df = df.dropna(axis=1, how='all')

    return df


def old_get_datasets():
    return ['bach', 'concrete', 'housing', 'pumadyn-8nh', 'servo']


def old_get_small_datasets():
    return ['bach', 'concrete', 'housing', 'servo']


def get_datasets():
    return get_small_datasets() + get_medium_datasets() + get_big_datasets()

def get_small_datasets():
    return ['challenger', 'fertility', 'concreteslump', 'autos', 'servo',
     'breastcancer', 'machine', 'yacht', 'autompg', 'housing', 'forest',
     'stock', 'pendulum', 'energy']


def get_medium_datasets():
    return ['concrete', 'solar', 'airfoil',
     'wine', 'gas', 'skillcraft', 'sml', 'parkinsons', 'pumadyn32nm']


def get_big_datasets():
    return ['pol', 'elevators', 'bike', 'kin40k', 'protein', 'tamielectric',
     'keggdirected', 'slice', 'keggundirected', '3droad', 'song',
     'buzz', 'houseelectric']


def shuffle(dataset: pd.DataFrame, seed=123456):
    """Helper method to shuffle a dataset dataframe"""
    indices = np.arange(len(dataset))
    s = np.random.get_state()
    np.random.seed(seed)
    np.random.shuffle(indices)
    np.random.set_state(s)
    return dataset.iloc[indices]


def mean_squared_error(y_pred: torch.Tensor, y_true: torch.Tensor):
    pass
    return ((y_pred - y_true)**2).mean().item()


def run_experiment(training_routine: Callable,
                   training_options: Dict,
                   dataset: Union[str, pd.DataFrame],
                   split: float,
                   cv: bool,
                   addl_metrics: Dict={},
                   repeats=1,
                   error_repeats=10,
                   ):
    """Main function to run a model on a dataset.

    This function is intended to take a training routine (implicitly instantiates
    a model and trains it),
    options for said training routine, any metrics that should be run
    in addition to negative log likelihood (if applicable) and mean squared
    error given in {'name': function} format, a dataset name/dataframe,
    a fraction determining the train/test split of the data, a bool that
    determines whether full CV is used (or whether a single train/test split
    is used), and finally how many times to repeat each fitting/evaluating
    step in each fold.

    The output is a dataframe of the results.

    Note that if the dataset if provided, it is assumed to be sufficiently
    shuffled already.

    :param training_routine:
    :param training_options:
    :param addl_metrics:
    :param dataset:
    :param split:
    :param cv:
    :return: results_list
    """

    # TODO: should this output a dataframe instead of a list? Dataframe might be nicer

    # if isinstance(dataset, str):
    #     dataset = old_load_dataset(dataset)
    #     dataset = shuffle(dataset)
    if isinstance(dataset, str):
        dataset = load_dataset(dataset)

    cols = list(dataset.columns)
    features = [x for x in cols if (x != 'target' and x.lower() != 'index')]

    n_per_fold = int(np.ceil(len(dataset)*split))
    n_folds = int(round(1/split))

    results_list = []

    for fold in range(n_folds):
        if not cv and fold > 0:  # only do one fold if you're not doing CV
            break
        train = dataset.iloc[:n_per_fold*fold]  # if fold=0, none before fold
        test = dataset.iloc[n_per_fold*fold:n_per_fold*(fold+1)]
        train = pd.concat([train,
                           dataset.iloc[n_per_fold*(fold+1):]])
        succeed = False
        n_errors = 0
        while not succeed and n_errors < error_repeats:
            try:
                trainX = torch.tensor(train[features].values, dtype=torch.float)
                trainY = torch.tensor(train['target'].values, dtype=torch.float)
                testX = torch.tensor(test[features].values, dtype=torch.float)
                testY = torch.tensor(test['target'].values, dtype=torch.float)

                for repeat in range(repeats):
                    result_dict = {'fold': fold,
                                   'repeat': repeat,
                                   'n': len(dataset),
                                   'd': len(features)-2}

                    start = time.perf_counter()
                    model_metrics, ypred = training_routine(trainX, trainY, testX,
                                                            testY,
                                                            **training_options)
                    end = time.perf_counter()

                    result_dict['mse'] = mean_squared_error(ypred, testY)
                    result_dict['train_time'] = end - start

                    # e.g. -ll, -mll
                    for name, value in model_metrics.items():
                        result_dict[name] = value

                    # e.g. mae, ...
                    for name, fxn in addl_metrics.items():
                        result_dict[name] = fxn(ypred, testY)
                    results_list.append(result_dict)
                    print('{} - {}'.format(datetime.datetime.now(), result_dict))
                    succeed = True
                    # print("succeed: ", succeed)
            except Exception:
                result_dict = dict(error=traceback.format_exc(),
                              fold=fold,
                              n=len(dataset),
                              d=len(features)-2)
                print(result_dict)
                traceback.print_exc()
                results_list.append(result_dict)
                n_errors += 1
                print('errors: ', n_errors)


    results = pd.DataFrame(results_list)
    return results


def run_experiment_suite(datasets,
                         training_routine: Callable,
                         training_options: Dict,
                         split: float,
                         cv: bool,
                         addl_metrics: Dict={},
                         inner_repeats=1,
                         outer_repeats=1,
                                            ):
    df = pd.DataFrame()
    if datasets == 'small':
        datasets = get_small_datasets()
    elif datasets == 'medium':
        datasets = get_medium_datasets()
    elif datasets == 'big':
        datasets = get_big_datasets()
    elif datasets == 'smalltomedium':
        datasets = get_small_datasets() + get_medium_datasets()
    elif datasets == 'all':
        datasets = get_datasets()
    else:
        raise ValueError("Unknown set of datasets")

    for i in range(outer_repeats):
        for dataset in datasets:
            print(dataset, 'starting...')
            result = run_experiment(training_routine, training_options, dataset=dataset,
                           split=split, cv=cv, addl_metrics=addl_metrics,
                           repeats=inner_repeats)
            result['dataset'] = dataset
            df = pd.concat([df, result])
            df.to_csv('./_partial_result.csv')

    return df


def train_SE_gp(trainX, trainY, testX, testY, ard=False, optimizer='lbfgs',
                n_epochs=100, lr=0.1, verbose=False, patience=20, smooth=True,
                noise_prior=False):
    [n, d] = trainX.shape
    # regular Gaussian likelihood for regression problem
    if noise_prior:
        noise_prior_ = gpytorch.priors.SmoothedBoxPrior(1e-4, 10, sigma=0.01)
    else:
        noise_prior_ = None

    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=noise_prior_)

    # no lengthscale prior etc.
    if ard:
        ard_num_dims = d
    else:
        ard_num_dims = None

    kernel = gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims)
    kernel = gpytorch.kernels.ScaleKernel(kernel)

    if optimizer == 'lbfgs':
        optimizer_ = torch.optim.LBFGS
    elif optimizer == 'adam':
        optimizer_ = torch.optim.Adam
    elif optimizer == 'sgd':
        optimizer_ = torch.optim.SGD
    else:
        raise ValueError("Unknown optimizer {}".format(optimizer))

    model = ExactGPModel(trainX, trainY, likelihood, kernel)


    # regular marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # fit GP
    fit_gp_model(model, likelihood, trainX, trainY, optimizer=optimizer_,
                 lr=lr, n_epochs=n_epochs, verbose=verbose, gp_mll=mll,
                 patience=patience, smooth=smooth)

    model_metrics = dict()
    with torch.no_grad():
        model.train()  # consider prior for evaluation on train dataset
        likelihood.train()
        train_outputs = model(trainX)
        model_metrics['prior_train_nmll'] = -mll(train_outputs, trainY).item()
        

        model.eval()  # Now consider posterior distributions
        likelihood.eval()
        train_outputs = model(trainX)
        test_outputs = model(testX)
        model_metrics['train_nll'] = -likelihood(train_outputs).log_prob(trainY).item()
        model_metrics['test_nll'] = -likelihood(test_outputs).log_prob(testY).item()
        model_metrics['train_mse'] = mean_squared_error(train_outputs.mean, trainY)

    return model_metrics, test_outputs.mean

#
# def train_rp_gp(trainX, trainY, testX, testY, k, ard=False, activation=None,
#                 dist='gaussian', optimizer='lbfgs', n_epochs=100, lr=0.1,
#                 verbose=False, patience=20, smooth=True):
#     [n, d] = trainX.shape
#
#     X = torch.cat([trainX, testX], dim=0)
#     Xprime, W, b = rp.ELM(X, k, dist, activation)
#
#     trainXprime = Xprime[:n, :]
#     testXprime = Xprime[n:, :]
#     # regular Gaussian likelihood for regression problem
#     likelihood = gpytorch.likelihoods.GaussianLikelihood()
#
#     # no lengthscale prior etc.
#     if ard:
#         ard_num_dims = k
#     else:
#         ard_num_dims = None
#     kernel = gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims)
#     kernel = gpytorch.kernels.ScaleKernel(kernel)
#
#     if optimizer == 'lbfgs':
#         optimizer_ = torch.optim.LBFGS
#     elif optimizer == 'adam':
#         optimizer_ = torch.optim.Adam
#     elif optimizer == 'sgd':
#         optimizer_ = torch.optim.SGD
#     else:
#         raise ValueError("Unknown optimizer {}".format(optimizer))
#
#     model = ExactGPModel(trainXprime, trainY, likelihood, kernel)
#
#     # regular marginal log likelihood
#     mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
#
#     # fit GP
#     fit_gp_model(model, likelihood, trainXprime, trainY, optimizer=optimizer_,
#                  lr=lr, n_epochs=n_epochs, verbose=verbose, gp_mll=mll,
#                  patience=patience, smooth=smooth)
#     model.eval()
#     likelihood.eval()
#     mll.eval()
#
#     model_metrics = dict()
#     with torch.no_grad():
#         train_outputs = model(trainXprime)
#         test_outputs = model(testXprime)
#
#         model_metrics['train_nmll'] = -mll(train_outputs, trainY).item()
#         model_metrics['test_nmll'] = -mll(test_outputs, testY).item()
#         model_metrics['train_nll'] = -likelihood(train_outputs).log_prob(trainY).item()
#         model_metrics['test_nll'] = -likelihood(test_outputs).log_prob(testY).item()
#
#     return model_metrics, test_outputs.mean


def train_additive_rp_gp(trainX, trainY, testX, testY, k, J, ard=True,
                         activation=None, optimizer='lbfgs',
                         n_epochs=100, lr=0.1, verbose=False, patience=20,
                         smooth=True, noise_prior=False):
    [n, d] = trainX.shape

    # X = torch.cat([trainX, testX], dim=0)
    # Xprime, W, b = rp.ELM(X, k, dist, activation)

    # regular Gaussian likelihood for regression problem
    if noise_prior:
        noise_prior_ = gpytorch.priors.SmoothedBoxPrior(1e-4, 10, sigma=0.01)
    else:
        noise_prior_ = None

    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=noise_prior_)

    if J < 1:
        raise ValueError("J<1")
    if ard:
        ard_num_dims = k
    else:
        ard_num_dims = None

    kernels = []
    projs = []
    bs = [torch.zeros(k)]*J
    for j in range(J):
        projs.append(rp.gen_rp(d, k))  # d, k just output dimensions of matrix
        kernels.append(gpytorch.kernels.RBFKernel(ard_num_dims))

    kernel = RPKernel(J, k, d, kernels, projs, bs, activation=activation)
    kernel = gpytorch.kernels.ScaleKernel(kernel)

    if optimizer == 'lbfgs':
        optimizer_ = torch.optim.LBFGS
    elif optimizer == 'adam':
        optimizer_ = torch.optim.Adam
    elif optimizer == 'sgd':
        optimizer_ = torch.optim.SGD
    else:
        raise ValueError("Unknown optimizer {}".format(optimizer))

    model = ExactGPModel(trainX, trainY, likelihood, kernel)

    # regular marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # fit GP
    fit_gp_model(model, likelihood, trainX, trainY, optimizer=optimizer_,
                 lr=lr, n_epochs=n_epochs, verbose=verbose, gp_mll=mll,
                 patience=patience, smooth=smooth)
    model.eval()
    likelihood.eval()
    mll.eval()

    model_metrics = dict()
    with torch.no_grad():
        model.train()  # consider prior for evaluation on train dataset
        likelihood.train()
        train_outputs = model(trainX)
        model_metrics['prior_train_nmll'] = -mll(train_outputs, trainY).item()
        

        model.eval()  # Now consider posterior distributions
        likelihood.eval()
        train_outputs = model(trainX)
        test_outputs = model(testX)
        model_metrics['train_nll'] = -likelihood(train_outputs).log_prob(trainY).item()
        model_metrics['test_nll'] = -likelihood(test_outputs).log_prob(testY).item()
        model_metrics['train_mse'] = mean_squared_error(train_outputs.mean, trainY)


    return model_metrics, test_outputs.mean


# def train_ELM(trainX, trainY, testX, testY, k, activation=None, optimizer='lbfgs',
#               n_epochs=1000, lr=0.1, verbose=False, patience=10):
#

def train_lr(trainX, trainY, testX, testY, optimizer='lbfgs',
              n_epochs=1000, lr=0.1, verbose=False, patience=10):
    model = LinearRegressionModel(trainX, trainY)

    loss = torch.nn.MSELoss()

    if optimizer == 'lbfgs':
        optimizer_ = torch.optim.LBFGS
    elif optimizer == 'adam':
        optimizer_ = torch.optim.Adam
    elif optimizer == 'sgd':
        optimizer_ = torch.optim.SGD
    else:
        raise ValueError("Unknown optimizer {}".format(optimizer))
    fit_linear_model(model, trainX, trainY, loss, optimizer_, lr, n_epochs, verbose, patience)

    model.eval()

    model_metrics = dict()

    ypred = model(testX)

    return model_metrics, ypred


def rp_ablation(split, cv, outer_repeats=1, inner_repeats=1):
    df = pd.DataFrame()
    for i in range(outer_repeats):
        for dataset in old_get_small_datasets():
            print(dataset, 'starting...')
            for x in range(13):
                k = 2**x
                try:
                    result = run_experiment(train_rp_gp,
                                            dict(verbose=False, lr=1.5,
                                                 optimizer='adam', n_epochs=2000,
                                                 k=k, activation=None, patience=20),
                                            dataset, split, cv, repeats=inner_repeats)
                    result['n_projections'] = k
                    result['dataset'] = dataset

                except Exception:
                    tb = traceback.format_exc()
                    result = pd.DataFrame([{'n_projections': k,
                                            'dataset': dataset,
                                            'error': tb}])

                df = pd.concat([df, result])
                df.to_csv('./_partial_ablation_results.csv')
    return df


def rp_compare_ablation(filename, fit=True, ard=False,
                        dsets=get_small_datasets()+get_medium_datasets(),
                        optimizer='lbfgs', lr=0.1, patience=10, verbose=False,
                        include_se=True, repeats=1, noise_prior=False):
    df = pd.DataFrame()

    if fit:
        epochs = 1000
        fname = './fitted_scaled_rp_compare_ablation_results.csv'
    else:
        epochs = 0
        fname = './unfitted_scaled_rp_compare_ablation_results.csv'
    if filename is not None:
        fname = filename


    for dataset in dsets:
        print(dataset, 'starting')

        if include_se:
            # regular SE kernel
            options = dict(verbose=verbose, lr=lr, optimizer=optimizer,
                           n_epochs=epochs, patience=patience, smooth=True,
                           ard=ard, noise_prior=noise_prior)
            options_json = json.dumps(options)
            try:

                result = run_experiment(train_SE_gp, options, dataset=dataset,
                                        split=0.1, cv=True, repeats=repeats)
                result['RP'] = False
                result['dataset'] = dataset
                result['options'] = options_json
            except Exception:
                result = dict()
                result['RP'] = False
                result['dataset'] = dataset
                result['error'] = traceback.format_exc()
                result['options'] = options_json
                print(result)
                result = pd.DataFrame([result])
            df = pd.concat([df, result])
            df.to_csv(fname)

        for k in [1, 4, 10]:
            for J in [1, 2, 3, 5, 8, 13, 20]:
                options = dict(verbose=False, ard=ard, activation=None,
                                optimizer=optimizer, n_epochs=epochs,
                                lr=lr, patience=patience, k=k, J=J,
                                smooth=True, noise_prior=noise_prior)
                options_json = json.dumps(options)
                try:
                    result = run_experiment(train_additive_rp_gp, options,
                                            dataset=dataset, split=0.1, cv=True,
                                            repeats=repeats)
                    result['RP'] = True
                    result['k'] = k
                    result['J'] = J
                    result['dataset'] = dataset
                    result['options'] = options_json
                except Exception:
                    result = dict()
                    result['RP'] = True
                    result['k'] = k
                    result['J'] = J
                    result['dataset'] = dataset
                    result['error'] = traceback.format_exc()
                    result['options'] = options_json
                    print(result)
                    result = pd.DataFrame([result])
                df = pd.concat([df, result])
                df.to_csv(fname)


if __name__ == '__main__':

    dsets = ['autos', 'fertility', 'energy', 'yacht']
    with gpytorch.settings.fast_computations(covar_root_decomposition=False,
                                             log_prob=False):
        rp_compare_ablation(fit=True,
                            filename='./tmp.csv',
                            dsets=dsets)


