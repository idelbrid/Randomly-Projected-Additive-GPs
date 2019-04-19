#!/bin/bash
# export PATH=/share/apps/anaconda3/5.2.0/bin:$PATH
# source activate ian-gpytorch-env
PYTHON="/home/iad35/.conda/envs/ian-gpytorch-env/bin/python"
$PYTHON gp_experiment_runner.py -m ./model_specs/RBF_model_spec.json -d pol elevators bike kin40k protein -o run_outputs/RBF_test_run2.csv -s 0.1 -r 2 --device cuda >& run_outputs/test_run_stdout.txt 
$PYTHON gp_experiment_runner.py -m ./model_specs/ARD_model_spec.json -d pol elevators bike kin40k protein -o run_outputs/ARD_test_run2.csv -s 0.1 -r 2 --device cuda  >& run_outputs/test_run_stdout.txt 

