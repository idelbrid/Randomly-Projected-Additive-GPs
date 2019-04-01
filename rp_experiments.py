import pandas as pd
import numpy as np
import torch
from typing import Callable, Dict, Union
import time
import datetime
import traceback
import os
from scipy.io import loadmat
import json
import gpytorch

from gp_helpers import mean_squared_error
from training_routines import train_SE_gp, train_additive_rp_gp, train_svi_gp


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


def format_timedelta(delta):
    d = delta.days
    h = delta.seconds // (60*60)
    m = (delta.seconds - h*(60*60)) // 60
    s = (delta.seconds - h*(60*60) - m*60)
    return '{}d {}h {}m {}s'.format(d, h, m, s)


def run_experiment(training_routine: Callable,
                   training_options: Dict,
                   dataset: Union[str, pd.DataFrame],
                   split: float,
                   cv: bool,
                   addl_metrics: Dict={},
                   repeats=1,
                   error_repeats=10,
                   normalize_using_train=True,
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

    if isinstance(dataset, str):
        dataset = load_dataset(dataset)

    cols = list(dataset.columns)
    features = [x for x in cols if (x != 'target' and x.lower() != 'index')]

    n_per_fold = int(np.ceil(len(dataset)*split))
    n_folds = int(round(1/split))

    results_list = []
    t0 = time.time()
    for fold in range(n_folds):
        if not cv and fold > 0:  # only do one fold if you're not doing CV
            break
        train = dataset.iloc[:n_per_fold*fold]  # if fold=0, none before fold
        test = dataset.iloc[n_per_fold*fold:n_per_fold*(fold+1)]
        train = pd.concat([train,
                           dataset.iloc[n_per_fold*(fold+1):]])
        if normalize_using_train:
            mu = train[features + ['target']].mean()

            train[features + ['target']] -= mu
            test[features + ['target']] -= mu
            for f in features + ['target']:
                sigma = train[f].std()
                if sigma > 0:
                    train[f] /= sigma
                    test[f] /= sigma
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
                                   'd': len(features)}

                    start = time.perf_counter()
                    ret = training_routine(trainX, trainY, testX,
                                                            testY,
                                                            **training_options)
                    model_metrics = ret[0]
                    ypred = ret[1]
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
                    succeed = True
                    t = time.time()
                    elapsed = t - t0
                    num_finished = fold*repeats+repeat+1
                    num_remaining = n_folds*repeats - num_finished
                    eta = datetime.timedelta(seconds=elapsed/num_finished*num_remaining)
                    print('{}, fold={}, rep={}, eta={} \n{}'.format(datetime.datetime.now(), fold, repeat, format_timedelta(eta), result_dict))

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


def rp_compare_ablation(filename, se_options, rp_options, fit=True,
                        dsets=get_small_datasets()+get_medium_datasets(),
                        optimizer='lbfgs', include_se=True, repeats=1):
    df = pd.DataFrame()

    if fit:
        epochs = 1000
        fname = './fitted_scaled_rp_compare_ablation_results.csv'
    else:
        epochs = 0
        fname = './unfitted_scaled_rp_compare_ablation_results.csv'
    if filename is not None:
        fname = filename

    se_options['k'] = 0
    se_options['J'] = 0


    for dataset in dsets:
        print(dataset, 'starting')

        if include_se:
            # regular SE kernel
            options_json = json.dumps(se_options)
            try:

                result = run_experiment(train_SE_gp, se_options, dataset=dataset,
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
                rp_options['k'] = k
                rp_options['J'] = J
                options_json = json.dumps(rp_options)
                try:
                    result = run_experiment(train_additive_rp_gp, rp_options,
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
    def run():
       with gpytorch.settings.fast_pred_var(True):
            run_experiment(train_additive_rp_gp, dict(verbose=False, ard=False, activation=None,
                                                      optimizer='adam', n_epochs=100, lr=0.1, patience=20, k=1, J=1,
                                                      smooth=True, noise_prior=True, ski=True, grid_ratio=1.0),
                           'sml', 0.1, cv=False)

