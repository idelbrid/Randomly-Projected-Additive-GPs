from math import pi

import numpy as np
from torch import __init__


def stybtang(x: torch.Tensor):
    return 1/2 * torch.sum(x.pow(4) - 16*x.pow(2) + 5*x, dim=-1)


def np_stybtang(x: np.array):
    return 1/2 * np.sum(np.power(x, 4) - 16*np.power(x, 2) + 5 * x, axis=-1)


def michalewicz(x: torch.Tensor, m: int):
    d = x.shape[-1]
    scaled_x = x * torch.arange(1, d+1, dtype=torch.float).unsqueeze(0)
    return -torch.sum(torch.sin(x) * torch.sin(scaled_x / pi).pow(2*m), dim=-1)


def np_michalewicz(x: np.array, m: int):
    d = x.shape[-1]
    scaled_x = x * np.arange(1, d+1, dtype=np.float)
    return -np.sum(np.sin(x) * np.power(np.sin(scaled_x / pi), 2*m), axis=-1)


def mixture_of_gaussians(x: torch.Tensor, mixtures, degree, sigma):
    shape = x.shape
    d = shape[-1]
    toreturn = torch.zeros(*shape[:-1])
    for i in range(mixtures):
        thisdeg = min(degree, d-i)
        mu = torch.zeros(degree)
        mu[0] = 1
        parts = x.index_select(dim=-1, index=torch.tensor([idx for idx in range(i, i+thisdeg)]))
        sqdist = torch.norm(parts - mu, dim=-1).pow(2)
        gauss = torch.exp(-1/(2*sigma**2) * sqdist) * i
        toreturn += gauss
    return -toreturn


def random_branin(x: torch.Tensor):
    d = x.shape[-1]
    torch.random.manual_seed(123456)
    i = torch.randint(0, d)
    j = torch.randint(0, d-1)
    if j >= i:  # easy way to choose 2 numbers w/o replacement
        j += 1
    return branin(x.index_select(dim=-1, index=torch.tensor([i,j])))


def branin(x: torch.Tensor):
    x1 = x.index_select(-1, torch.tensor(0))
    x2 = x.index_select(-1, torch.tensor(1))
    comp1 = (x2 - 5.1 / (4 * pi**2) * x1.pow(2) + 5/pi * x1 - 6).pow(2)
    comp2 = 10*(1-1/(8*pi))*torch.cos(x1) + 10
    return comp1 + comp2


def easy_meshgrid(sizes, numpy=False, interior=True):
    spot_per_dim = []
    for i in range(len(sizes)):
        if interior:
            spots = torch.linspace(0, 1, sizes[i] + 2)[1:-1]
        else:
            spots = torch.linspace(0, 1, sizes[i])
        spot_per_dim.append(spots)
    tensors = torch.meshgrid(spot_per_dim)
    stacked = torch.stack(tensors)
    res = stacked.reshape(len(sizes), -1).t()
    if numpy:
        res = res.numpy()
    return res