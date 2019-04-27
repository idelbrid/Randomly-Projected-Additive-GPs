import numpy as np
import torch
from math import sqrt
from torch.distributions import Categorical

def gen_rp(d, k, dist='gaussian'):
    """Generate a random projection matrix (input dim d output dim k)"""
    if dist == 'gaussian':
        return torch.randn(d, k) / np.sqrt(k)
    elif dist == 'sphere':
        W = torch.randn(d, k)
        vecnorms = torch.norm(W, p=2, dim=0, keepdim=True)
        W = torch.div(W, vecnorms)
        # variance of w drawn uniformly from unit sphere is
        # 1/d
        return W * sqrt(d) / sqrt(k)
    elif dist == 'very-sparse':
        categorical = Categorical(torch.tensor([1/(2 * sqrt(d)), 1-1/sqrt(d), 1/(2*sqrt(d))]))
        samples = categorical.sample(torch.Size([d, k])) - 1
        samples = samples.to(dtype=torch.float)
        return samples
    elif dist == 'bernoulli':
        return (torch.bernoulli(torch.rand(d, k)) * 2 - 1) / sqrt(k)
    elif dist == 'uniform':
        # variance of uniform on -1, 1 is 1 / 3
        return (torch.rand(d, k) * 2 - 1) / sqrt(k) * sqrt(3)
    else:
        raise ValueError("Not a valid RP distribution")


def gen_pca_rp(d, k, W, D):
    """
    :param d: dimension of data
    :param k: dimension of projection
    :param W: eigenvectors (as p x d matrix)
    :param D: eigenvalues (as p x 1 vector)
    :return:
    """
    p = W.shape[0]
    if not p == D.shape[0] and d >= p:
        raise ValueError("dimension didn't make sense. Check orientation of eigenvectors or eigenvalues")
    distribution = torch.distributions.MultivariateNormal(torch.zeros(p),
                                                           covariance_matrix=torch.diag(D))
    samples = distribution.rsample([k])  # k x p
    samples = samples.matmul(W)  # (k x p)*(p x d) = k x d
    samples = samples.t()  # make it match gen_rp (input dim x output dim)
    vecnorms = torch.norm(samples, p=2, dim=0, keepdim=True)
    samples = torch.div(samples, vecnorms)
    return samples * sqrt(d) / sqrt(k)


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


def space_equally(P, lr, niter):
    P.requires_grad = True
    n, d = P.shape

    def loss(P):
        ones = torch.ones(n).unsqueeze(0)
        otherones = torch.ones(d).unsqueeze(0)
        norms = torch.sqrt(torch.pow(P, 2).matmul(otherones.t()))  # should be n x 1
        norm_products = norms.matmul(norms.t())  # matrix of norm products
        outer = P.matmul(P.t())  # matrix of dot products
        cosine = torch.div(outer, norm_products)  # matrix of cos(angle) between vectors
        cosine = cosine - torch.eye(n)
        # angle = torch.acos(cosine)
        square = torch.pow(cosine, 4)  # square it to make it sign invariant and differentiable
        summation = ones.matmul(square).matmul(ones.t()) # sum all entries
        cost = summation
        return summation, cost

    for i in range(niter):
        summation, cost = loss(P)
        # if i % 100 == 0:
        #     print(i, summation)
        cost.backward()
        P.data.sub_(lr*P.grad.data)
        P.grad.zero_()
    summation, cost = loss(P)

    otherones = torch.ones(d).unsqueeze(0)
    norms = torch.sqrt(torch.pow(P, 2).matmul(otherones.t()))  # should be n x 1
    P.data.div_(norms.data)
    P.requires_grad = False
    return P, summation


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
