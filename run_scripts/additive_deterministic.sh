#!/bin/bash
# export PATH=/share/apps/anaconda3/5.2.0/bin:$PATH
# source activate ian-gpytorch-env
PYTHON="/home/iad35/.conda/envs/ian-gpytorch-env/bin/python"

$PYTHON gp_experiment_runner.py -m ./model_specs/GAM_spec.json -d orientation400 orientation1200 orientation3600 -o run_outputs/graphite_GAM_unweighted_mem_efficient_orientations.csv -s 0.1 -r 2 --skip_posterior_variances --cg_tol 0.002 --eval_cg_tol 0.001 --skip_evaluate_on_train --skip_random_restart --memory_efficient --device cuda:0,cuda:1,cuda:2 --no_toeplitz --error_repeats 3 >& run_outputs/GAM_uw_mem_eff_orientations_stdout.txt

