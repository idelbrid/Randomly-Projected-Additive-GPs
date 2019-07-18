#!/bin/bash
# export PATH=/share/apps/anaconda3/5.2.0/bin:$PATH
# source activate ian-gpytorch-env
PYTHON="/home/iad35/.conda/envs/ian-gpytorch-env/bin/python"

$PYTHON gp_experiment_runner.py -m ./model_specs/additive_spread_prescale_Jd.json -d orientation400 orientation1200 orientation3600 -o run_outputs/graphite_additive_spread_prescale_Jd_orientations.csv -s 0.1 -r 2 --skip_posterior_variances --skip_evaluate_on_train --skip_random_restart --memory_efficient --cg_tol 0.002 --eval_cg_tol 0.001 --device cuda:0,cuda:1,cuda:2 --error_repeats 3 >& run_outputs/additive_spread_prescale_stdout_3gpu_orientations_no_ablation.txt 


# $PYTHON gp_experiment_runner.py -m ./model_specs/additive_spread_prescale_Jd.json -d pumadyn32nm -o run_outputs/full_dpagp_ard_pumadyn_mutligpu.csv -s 0.1 -r 2 --skip_posterior_variances --skip_evaluate_on_train --skip_random_restart --cg_tol 0.002 --eval_cg_tol 0.001 --memory_efficient --device cuda:0,cuda:1 --error_repeats 3 >& run_outputs/full_dpagp_ard_puma_multigpu_no_RR.txt 

# $PYTHON gp_experiment_runner.py -m ./model_specs/additive_spread_prescale_Jd.json -d pol elevators bike kin40k protein -o run_outputs/full_dpagp_ard_pol-to-protein_checkpoint.csv -s 0.1 -r 2 --skip_posterior_variances --skip_evaluate_on_train --skip_random_restart --cg_tol 0.002 --eval_cg_tol 0.001 --memory_efficient --device cuda:0,cuda:1 --checkpoint_kernel 8000 --error_repeats 3 >& run_outputs/full_dpagp_ard_pol-to-protein_checkpoint.txt 

# CUDA_LAUCH_BLOCKING=1 $PYTHON -m cProfile -o mem_efficient_dpagp2.prof gp_experiment_runner.py -m ./model_specs/additive_spread_prescale_Jd.json -d pol -o run_outputs/test_add_spread_ard_Jd.csv -s 0.1 --no_cv --skip_posterior_variances --skip_evaluate_on_train --skip_log_det_forward --cg_tol 0.002 --eval_cg_tol 0.001 --no_toeplitz --device cuda --error_repeats 1 >& run_outputs/test_add_spread_ard_Jd.txt

# CUDA_LAUCH_BLOCKING=1 $PYTHON -m cProfile -o nonmem_efficient_dpagp.prof gp_experiment_runner.py -m ./model_specs/additive_spread_prescale_Jd.json -d pol -o run_outputs/test_add_spread_ard_Jd.csv -s 0.1 --no_cv --skip_posterior_variances --skip_evaluate_on_train --skip_log_det_forward --cg_tol 0.002 --eval_cg_tol 0.001 --no_toeplitz --device cuda --checkpoint_kernel 1500 --error_repeats 1 >& run_outputs/test_add_spread_ard_Jd2.txt


# $PYTHON gp_experiment_runner.py -m ./model_specs/additive_spread_prescale_Jd_ski.json -d pumadyn32nm -o run_outputs/test_add_spread_ard_Jd_ski.csv -s 0.1 --no_cv --skip_posterior_variances --skip_evaluate_on_train --cg_tol 0.002 --eval_cg_tol 0.001 --no_toeplitz --device cuda --error_repeats 1 >& run_outputs/test_add_spread_ard_Jd_ski.txt
