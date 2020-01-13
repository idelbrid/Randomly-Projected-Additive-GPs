# Randomly Projected Additive GPs git repo

This repo contains implementations and experiment code for the paper [Randomly Projected Additive Gaussian Processes for Regression](https://arxiv.org/abs/1912.12834)

# Requirements
- Python > 3.0
- GPyTorch >= 1.0
- PyKeOps >= 1.2


## Files

* `config_template.py`: Template configuration file for dataset file locations, etc. **Rename to `config.py` and replace with your file configurations**.  UCI datasets referenced in the experiments may be [downloaded here](https://drive.google.com/file/d/0BxWe_IuTnMFcYXhxdUNwRHBKTlU/view).
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

## UCI Data Sets
To download the UCI data sets used for benchmarks, download them from [Andrew Gordon Wilson's home page](https://people.orie.cornell.edu/andrew/pattern/#Data). See `config_template.py` for details on how these files are expected to be organized in accordance with your configurations.
