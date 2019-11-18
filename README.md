# Randomly Projected Additive GPs git repo

This repo contains implementations and experiment code for the paper <b>\<TODO: ADD PAPER NAME\></b>. 

## Files

* `config_template.py`: Template configuration file for dataset file locations, etc. 
* `gp_experiment_runner.py`: Command-line endpoint used for running batches of experiments. 
* `synthetic_test_script.py`: A simple script for running synthetic experiments.
* `rp.py`: Generating (random) projection matrices, including a routine for generating diversified projection matrices (useed in DPA-GP).
* `training_routines.py`: A collection of routines used to construct, train, and test GPs in this project. 
* `test.py`: a suite of unit tests.
* `utils.py`: Utilities that are reused and don't live in a particular section of the project.

## Subpackages
* `gp_models`: Encapsulates the model (and kernel) definitions for kernels and models used.
* `fitting`: Encapsulates methods for learning. Currently, only optimization-based methods are available, as opposed to, e.g., sampling.

## Folders
* `model_specs`: Model specification .json files. These are used to store and re-use the configuration of models.
* `run_scripts`: Re-used/example command-line calls to `gp_experiment_runner.py`.

