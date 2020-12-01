import numpy as np
from numpy.random import random

import math

def exprnd(alpha, n=1):
    n = int(n)
    U = random(n)
    return -np.log(1 - U) / alpha, U

def cauchy_rnd(a, b, n=1):
    Y = random(n)
    return a + b * np.tan(np.pi * (Y - 0.5)), Y


# returns 2 lists of pairs (X1, X2)
# X1 = sqrt(Y1) * cos(Y2)
# X2 = sqrt(Y1) * sin(Y2)
def N_polarrnd(n=1):
    n = int(n)
    Y1 = exprnd(0.5, n)[0]
    Y2 = 2 * np.pi * random(n)
    return np.sqrt(Y1) * np.cos(Y2), np.sqrt(Y1) * np.sin(Y2)


from generators_discrete import bernrnd
def N01rnd(n):
    n = int(n)
    def p(x):
        return np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)
    def q(x):
        return 1. / np.pi / (x ** 2 + 1)
    
    k = np.sqrt(2 * np.pi / np.exp(1))
    Y = np.zeros(n)
     
    X = []
    n_cur = 0
    while n_cur != n:
        x = cauchy_rnd(0, 1, 1)[0][0]
        Y = bernrnd(p(x) / k / q(x))
        if Y:
            X.append(x)
        n_cur += Y
    return np.array(X)