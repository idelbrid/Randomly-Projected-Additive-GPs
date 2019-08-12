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
from config import data_base_path

from fitting.optimizing import mean_squared_error
import training_routines


def old_load_dataset(name: str):
    """Helper method to load a given dataset by name"""
    return pd.read_csv('./datasets/{}/dataset.csv'.format(name))


def load_dataset(name: str):
    mat = loadmat(os.path.join(data_base_path, 'uci', name, '{}.mat'.format(name)))
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


def _determine_folds(split, dataset):
    """Determine the indices where folds begin and end."""
    n_per_fold = int(np.floor(len(dataset) * split))
    n_folds = int(round(1 / split))
    remaining = len(dataset) - n_per_fold * n_folds
    fold_starts = [0]
    for i in range(n_folds):
        if i < remaining:
            fold_starts.append(fold_starts[i] + n_per_fold + 1)
        else:
            fold_starts.append(fold_starts[i] + n_per_fold)
    return fold_starts


def _access_fold(dataset, fold_starts, fold):
    """Pull out the test and train set of a dataset using existing fold division"""
    train = dataset.iloc[0:fold_starts[fold]]  # if fold=0, none before fold
    test = dataset.iloc[fold_starts[fold]:fold_starts[fold + 1]]
    train = pd.concat([train, dataset.iloc[fold_starts[fold + 1]:]])
    return train, test


def _normalize_by_train(train, test):
    """Mean and std normalize using mean and std of the train set."""
    train = train.copy()
    test = test.copy()
    cols = list(train.columns)
    features = [x for x in cols if (x != 'target' and x.lower() != 'index')]
    mu = train[features + ['target']].mean()

    train.loc[:, features + ['target']] -= mu
    test.loc[:, features + ['target']] -= mu
    for f in features + ['target']:
        sigma = train[f].std()
        if sigma > 0:
            train.loc[:, f] /= sigma
            test.loc[:, f] /= sigma
    return train, test


def run_experiment(training_routine: Callable,
                   training_options: Dict,
                   dataset: Union[str, pd.DataFrame],
                   split: float,
                   cv: bool,
                   addl_metrics: Dict={},
                   repeats=1,
                   error_repeats=10,
                   normalize_using_train=True,
                   chosen_fold=0,
                   print_to_console=True
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

    fold_starts = _determine_folds(split, dataset)
    n_folds = len(fold_starts) - 1

    results_list = []
    t0 = time.time()
    for fold in range(n_folds):
        if not cv and fold != chosen_fold:  # only do one fold if you're not doing CV
            continue
        train, test = _access_fold(dataset, fold_starts, fold)
        if normalize_using_train:
            train, test = _normalize_by_train(train, test)
        succeed = False
        n_errors = 0
        while not succeed and n_errors < error_repeats:
            try:
                trainX = torch.tensor(train[features].values, dtype=torch.float).contiguous()
                trainY = torch.tensor(train['target'].values, dtype=torch.float).contiguous()
                testX = torch.tensor(test[features].values, dtype=torch.float).contiguous()
                testY = torch.tensor(test['target'].values, dtype=torch.float).contiguous()

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
                    result_dict['rmse'] = np.sqrt(result_dict['mse'])
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
                    if print_to_console:
                        print('{}, fold={}, rep={}, eta={} \n{}'.format(datetime.datetime.now(), fold, repeat, format_timedelta(eta), result_dict))

                    # print("succeed: ", succeed)
            except Exception:
                result_dict = dict(error=traceback.format_exc(),
                              fold=fold,
                              n=len(dataset),
                              d=len(features)-2,
                                   mse=np.nan, rmse=np.nan)
                print(result_dict)
                traceback.print_exc()
                results_list.append(result_dict)
                n_errors += 1
                print('errors: ', n_errors)

    results = pd.DataFrame(results_list)
    print('Mean RMSE = {}'.format(results['rmse'].mean()))
    return results


# def run_experiment_suite(datasets,
#                          training_routine: Callable,
#                          training_options: Dict,
#                          split: float,
#                          cv: bool,
#                          addl_metrics: Dict={},
#                          inner_repeats=1,
#                          outer_repeats=1,
#                                             ):
#     df = pd.DataFrame()
#     if datasets == 'small':
#         datasets = get_small_datasets()
#     elif datasets == 'medium':
#         datasets = get_medium_datasets()
#     elif datasets == 'big':
#         datasets = get_big_datasets()
#     elif datasets == 'smalltomedium':
#         datasets = get_small_datasets() + get_medium_datasets()
#     elif datasets == 'all':
#         datasets = get_datasets()
#     else:
#         raise ValueError("Unknown set of datasets")
#
#     for i in range(outer_repeats):
#         for dataset in datasets:
#             print(dataset, 'starting...')
#             result = run_experiment(training_routine, training_options, dataset=dataset,
#                            split=split, cv=cv, addl_metrics=addl_metrics,
#                            repeats=inner_repeats)
#             result['dataset'] = dataset
#             df = pd.concat([df, result])
#             df.to_csv('./_partial_result.csv')
#
#     return df


# def rp_compare_ablation(filename, datasets, rp_options, repeats=1, max_j=300):
#     if os.path.exists(filename):
#         df = pd.read_csv(filename)
#     else:
#         df = pd.DataFrame({'dataset': [], 'J': []})
#
#     for dataset in datasets:
#         print(dataset, 'starting')
#         J_2 = 0
#         J_1 = 1
#         J = J_1 + J_2
#         while J_1 + J_2 < max_j:
#             J = J_1 + J_2
#             J_2 = J_1
#             J_1 = J
#             print("J=", J)
#             if not df[(df['J'] == J) & (df['dataset'] == dataset)].empty:
#                 print("already done, skipping...")
#                 continue
#             rp_options['model_kwargs']['J'] = J
#             options_json = json.dumps(rp_options)
#             with gpytorch.settings.cg_tolerance(0.01):
#                 result = run_experiment(training_routines.train_exact_gp, rp_options,
#                         dataset=dataset, split=0.1, cv=True,
#                         repeats=repeats, normalize_using_train=True)
#             result['RP'] = True
#             result['k'] = rp_options['model_kwargs']['k']
#             result['J'] = rp_options['model_kwargs']['J']
#             result['dataset'] = dataset
#             result['options'] = options_json
#             df = pd.concat([df, result])
#             df.to_csv(filename)


if __name__ == '__main__':
    import json
    import argparse

    parser = argparse.ArgumentParser(description='Utility to run a suite of experiments with a GP model on UCI regression datasets.')
    parser.add_argument('-m', '--model_spec', type=str, help='path to model specification json file', required=True)
    parser.add_argument('-d', '--datasets', type=str, nargs='+', required=True, help='UCI dataset name(s) or predefined set of of datasets')
    parser.add_argument('-o', '--output', type=str, required=True, help='path to output csv file')
    parser.add_argument('-s', '--split', type=float, default=0.1, required=False, help='fraction of data in test set')
    parser.add_argument('-r', '--repeats', type=int, default=1, required=False, help='number of times to repeat each fold')
    parser.add_argument('--no_cv', action='store_false', dest='cv')
    parser.add_argument('--cg_tol', type=float, default=0.05, required=False)
    parser.add_argument('--eval_cg_tol', type=float, default=0.01, required=False)
    parser.add_argument('--fast_pred', dest='fast_pred', action='store_true')
    parser.add_argument('--use_chol', action='store_true')
    parser.add_argument('--no_toeplitz', dest='use_toeplitz', action='store_false')
    parser.add_argument('--memory_efficient', dest='memory_efficient', action='store_true')
    parser.add_argument('--device', type=str, default='cpu', required=False, help='device string to use in PyTorch')
    parser.add_argument('--skip_posterior_variances', action='store_true')
    parser.add_argument('--ablation', action='store_true')
    parser.add_argument('--J', type=int, nargs='+', required=False, help='Js to use in ablation to overwrite the ablation Js')
    parser.add_argument('--k', type=int, nargs='+', required=False, help='If used, do ablation on k (Dj) with given k values instead of J.')
    parser.add_argument('--fold', type=int, default=0, required=False)
    parser.add_argument('--error_repeats', type=int, default=10, required=False)
    parser.add_argument('--max_cg_iterations', type=int, default=10_000, required=False)
    parser.add_argument('--skip_evaluate_on_train', action='store_true')
    parser.add_argument('--skip_random_restart', action='store_true')
    parser.add_argument('--skip_log_det_forward', action='store_true', required=False, help='Apply skip log det forward option.')
    parser.add_argument('--checkpoint_kernel', type=int, default=0, required=False, help='Split kernel into chunks')
    parser.add_argument('--record_pred_unc', action='store_true', required=False, help='Record predictive uncertainty metrics.')
    parser.add_argument('--double', action='store_true', required=False, help='Run experiments in double precision rather than float.')

    args = parser.parse_args()

    print('Parser arguments', args)

    with open(args.model_spec, 'r') as f:
        options = json.load(f)

    print('Loaded options', options)

    devices = args.device.split(',')
    print('Using device(s) {}'.format(devices))

    print('Registered data base path {}'.format(data_base_path))

    try:
        args.datasets[0] = int(args.datasets[0])
    except Exception:
        pass  # assume not an int

    if len(args.datasets) == 1:
        if args.datasets[0] == 'all':
            datasets = get_datasets()
        elif args.datasets[0] == 'small':
            datasets = get_small_datasets()
        elif args.datasets[0] == 'small-med':
            datasets = get_datasets()[:18]  # through wine
        elif args.datasets[0] == 'med':
            datasets = get_datasets()[18:24]  # med through pol
        elif args.datasets[0] == 'large':
            datasets = get_datasets()[24:]
        elif isinstance(args.datasets[0], int):
            datasets = get_datasets()[:args.datasets[0]]
        else:
            datasets = args.datasets
    else:
        datasets = args.datasets

    # Disambiguate overloaded "kind" key word option
    ppr, cgp, ma = False, False, False
    if options['kind'] == 'ppr_gp':
        options.pop('kind')
        ppr = True
    elif options['kind'] == 'cgp':
        options.pop('kind')
        cgp = True
    elif options['kind'] == 'model_average':
        options.pop('kind')
        ma_args = options.pop('varying_params')
        options = options['base_model_kwargs']
        options['model_kwargs']['varying_params'] = ma_args
        ma = True
    else:
        # We're doing an exact GP
        options['skip_random_restart'] = args.skip_random_restart

    options['devices'] = devices
    options['skip_posterior_variances'] = args.skip_posterior_variances
    options['evaluate_on_train'] = not args.skip_evaluate_on_train
    options['record_pred_unc'] = args.record_pred_unc
    if args.double:
        options['double'] = args.double
        # Otherwise, leave it out for backwards compatability.

    if options['record_pred_unc'] and options['skip_posterior_variances']:
        raise ValueError("Can't record predictive uncertainty while skipping posterior variances.")


    df = pd.DataFrame()
    for dataset in datasets:
        print('Starting dataset {}'.format(dataset))
        with gpytorch.settings.cg_tolerance(args.cg_tol), \
              gpytorch.settings.eval_cg_tolerance(args.eval_cg_tol), \
              gpytorch.settings.fast_computations(not args.use_chol, not args.use_chol, not args.use_chol), \
              gpytorch.settings.fast_pred_var(args.fast_pred), \
              gpytorch.settings.use_toeplitz(args.use_toeplitz), \
              gpytorch.settings.max_cg_iterations(args.max_cg_iterations), \
              gpytorch.beta_features.checkpoint_kernel(args.checkpoint_kernel), \
              gpytorch.settings.skip_logdet_forward(args.skip_log_det_forward), \
              gpytorch.settings.memory_efficient(args.memory_efficient):
            if args.ablation:
                if args.k is not None:
                    abl_vars = args.k
                elif args.J is not None:
                    abl_vars = args.J
                else:
                    abl_vars = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
            else:
                abl_vars = [-1]

            for abl_val in abl_vars:
                if args.ablation:
                    if args.k is None:
                        options['model_kwargs']['J'] = abl_val
                    else:
                        options['model_kwargs']['k'] = abl_val  # TODO: check this is right.

                if ppr:
                    routine = training_routines.train_ppr_gp
                elif cgp:
                    routine = training_routines.train_compressed_gp
                elif ma:
                    routine = training_routines.train_exact_gp_model_average
                else:
                    routine = training_routines.train_exact_gp

                results = run_experiment(routine, options,
                               dataset, split=args.split, cv=args.cv, repeats=args.repeats,
                                         normalize_using_train=True, chosen_fold=args.fold,
                                         error_repeats=args.error_repeats)
                if args.ablation:
                    if args.k is None:
                        results['J'] = abl_val
                    else:
                        results['k'] = abl_val

                results['dataset'] = dataset
                results['options'] = json.dumps(options)
                results['cg_tol'] = args.cg_tol
                results['eval_cg_tol']= args.eval_cg_tol
                results['use_chol'] = args.use_chol
                results['max_cg_iterations'] = args.max_cg_iterations
                results['use_toeplitz'] = args.use_toeplitz
                results['fast_pred_var'] = args.fast_pred
                results['checkpoint_kernel'] = args.checkpoint_kernel
                results['skip_log_det_forward'] = args.skip_log_det_forward
                results['memory_efficient'] = args.memory_efficient
                df = pd.concat([df, results])
                df.to_csv(args.output)
