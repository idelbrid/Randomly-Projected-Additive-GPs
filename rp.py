import numpy as np
import torch


def np_gen_rp(k, d, dist='gaussian'):
    """Generate weight matrix W (k x d) for RP"""

    if dist == 'gaussian':
        return np.random.standard_normal([k, d]) / np.sqrt(k)
    elif dist == 'rademacher':
        return np.random.rand(k, d) > 0.5
    elif dist == 'uniform':
        return np.random.uniform(-1, 1, [k, d])
    else:
        raise ValueError("Not a valid RP distribution")


def gen_rp(k, d, dist='gaussian'):
    """Generate a random projection matrix (input dim k output dim d)"""
    if dist == 'gaussian':
        return torch.randn(k, d)/np.sqrt(d)
    elif dist == 'sphere':
        W = torch.randn(k, d)
        vecnorms = torch.norm(W, p=2, dim=0, keepdim=True)
        return torch.div(W, vecnorms)
    elif dist == 'bernoulli':
        return torch.bernoulli(torch.rand(k, d))
    elif dist == 'uniform':
        return torch.rand(k, d) * 2 - 1
    else:
        raise ValueError("Not a valid RP distribution")


def Sigmoid(a: torch.Tensor, b, x):
    return torch.sigmoid(x.matmul(a) + b)


def LinearProjection(a: torch.Tensor, b, x):
    return x.matmul(a)


def Tanh(a: torch.Tensor, b, x):
    return torch.tanh(x.matmul(a) + b)


def Gaussian(a: torch.Tensor, b, x):
    x = x.unsqueeze(2)
    a = a.unsqueeze(0)
    b = b.unsqueeze(0)
    # print('x shape', x.shape)
    # print('a shape', a.shape)
    # print('b shape', b.shape)
    diffs = x - a
    # print('diff shape', diffs.shape)
    norms = torch.norm(diffs, 2, dim=1)
    # print('norm shape', norms.shape)
    return (norms * b).squeeze()


def Multiquadratic(a, b, x):
    x = x.unsqueeze(2)
    a = a.unsqueeze(0)
    b = b.unsqueeze(0)
    return torch.sqrt(torch.norm(x - a, 2, dim=1) + b**2)


def Hard_limit(a, b, x):
    return (x.matmul(a) + b <= 0).float()


def Fourier(a, b, x):
    return torch.cos(x.matmul(a) + b)


def ELM(X, K, dist='gaussian', activation='sigmoid'):
    [n, d] = X.size()
    A = gen_rp(d, K, dist)
    b = gen_rp(1, K, dist)
    fn = None
    if activation is None:
        fn = LinearProjection
    elif callable(activation):
        fn = activation
    elif activation=='sigmoid':
        fn = Sigmoid
    elif activation=='tanh':
        fn = Tanh
    elif activation=='gaussian':
        fn = Gaussian
    elif activation=='multiquadratic':
        fn = Multiquadratic
    elif activation=='hard_limit':
        fn = Hard_limit
    elif activation=='fourier':
        fn = Fourier
    else:
        raise ValueError("Invalid activation")

    return fn(A, b, X), A, b



if __name__ == '__main__':
    # Basically normal RP
    n = 100
    d = 20
    k = 1000
    X = torch.rand(n, k)
    W = gen_rp(k, d, 'gaussian')

    Y = X.matmul(W)
    # print(Y.shape)

    # ELM style
    X = torch.rand(n, d)
    A = gen_rp(d, k, 'gaussian')
    b = gen_rp(1, k)
    nodes1 = Sigmoid(A, b, X)
    nodes2 = Tanh(A, b, X)
    nodes3 = Gaussian(A, b, X)
    nodes4 = Multiquadratic(A, b, X)
    nodes5 = Hard_limit(A, b, X)
    nodes6 = Fourier(A, b, X)
