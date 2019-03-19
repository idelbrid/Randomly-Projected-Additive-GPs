# Scalable GPs git repo

This repo is a work-in-progress for running experiments related to scalable Gaussian Processes. It is planned to contain code to (1) preprocess many benchmark datasets, (2) implement and run (scalable) Gaussian Process models, primarily using GPyTorch, (3) easily, modularly, and systematically run batches of experiments, and (4) visualize results.

## Files

* `rp.py`: code for generating random projection matrices (and some ELM code)
* `gp_helpers.py`: module containing
    * RP kernel implementation
    * Routine for training to convergence
    * GPyTorch model classes
* `training_routines.py`: routines for running GP models with different configurations
* `rp_experiments.py`: routines for handling datasets and running models on many datasets with or without cross-validation.


## Notebooks...

* Most are not permanent. Just fixtures to help run code and debug.
