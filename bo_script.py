import bayesopt as bo
import torch
import gpytorch
import gp_models
import gp_experiment_runner
import training_routines
import matplotlib.pyplot as plt
import numpy as np
import scipy
from math import pi
import pickle

traj = []
for j in range(20):
    ys = []
    best_ys = []
    d = 10
    bounds = [(-4, 4) for _ in range(d)]
    objective_function = bo.stybtang
    best_y = np.inf
    for iter_ in range(200):
        x = torch.rand(d)*8 - 4
        y = objective_function(x)
        ys.append(y.item())
        if y.item() < best_y:
            best_y = y.item()
        best_ys.append(best_y)
        # print(iter_, best_y)
    traj.append(best_ys)
with open('stybtang_random_samples.pkl', 'wb') as f:
    pickle.dump(traj, f)
traj = np.array(traj)
plt.plot(traj.mean(axis=0))
plt.fill_between(np.array(len(traj[0, :])), traj.min(axis=0), traj.max(axis=0), alpha=0.2)
plt.show()