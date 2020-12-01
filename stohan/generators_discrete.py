import numpy as np
from numpy.random import random

import math

def bernrnd(p,*size):
    if p>1 or p<0:
        error('p = %4.2f is out of bounds [0, 1]')
    return random(size) < p

def binrnd(n, p):
    if p>1 or p<0:
        error('p = %4.2f is out of bounds [0, 1]')
    return sum(random(n) < p)

def geomrnd(p,n):
    if p>1 or p<0:
        error('p = %4.2f is out of bounds [0, 1]')
    sucsess = np.zeros(n, dtype=bool);
    X = np.zeros(n+1);

    # пока есть тот, который не завершился успехом
    idx = np.arange(n+1);
    while len(idx) != 0:
        Y = bernrnd(p, n)
        sucsess |= Y
        idx = np.where(~sucsess)[0]
        X[idx] += 1
    return X

# def geomrnd1(p,n):
#     if p>1 or p<0:
#         error('p = %4.2f is out of bounds [0, 1]')
#     U = random(n)
#     xmax = int(math.ceil(np.log(1e-3) / np.log(1-p) - 1))
#     F = np.cumsum(np.cumprod(np.hstack((p, (1-p) * np.ones(2 * xmax))))

#     [row, col, ~] = find(U <= F.T);
#     X = np.zeros(n); 
#     for i = 1:n
#         X(i) = row(find(col == i, 1));
#     end
#     X = X - 1
#     return X

def kolmcdf(x):
    n = int(1e2)
    pows = -2 * (np.arange(1, n+1) ** 2) * (x**2)
    sign_ = np.cumprod(-np.ones(n))
    F = 1 + 2 * sum(sign_ * np.exp(pows))
    return F

# n - объем выборки, eps -  точность расчета
def cantrnd(n=1,eps=1e-10):
    m = int(1 - round(np.log(eps)/np.log(3)))
    bern = bernrnd(0.5,n,m)
    deg = -np.arange(1, m+1).reshape(-1, 1)
    X = np.sum(2 * bern @ (3. ** deg), axis=1)
    F = 0.5 + np.sum((bern - 0.5) @ (2. ** deg), axis=1)
    return X, F

from generators_continuous import exprnd

def poisrnd(rate, n):
    k = np.zeros(n)
    s, _ = exprnd(1, n)
    idx = np.arange(n)[s < rate]
    while len(idx) != 0:
        s += exprnd(1, n)[0]
        k[idx] += 1
        idx = np.arange(n)[s < rate]
    return k