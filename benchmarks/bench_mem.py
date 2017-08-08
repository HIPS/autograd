import autograd.numpy as np
from autograd import grad

def peakmem_needless_nodes():
    N, M = 1000, 100
    def fun(x):
        for i in range(M):
            x = x + 1
        return np.sum(x)

    grad(fun)(np.zeros((N, N)))
