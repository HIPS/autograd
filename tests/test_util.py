import numpy as np
import itertools as it
from funkyyak import grad

def nd(fun, args, argnum):
    args = list(args)
    def f(x):
        args[argnum] = x
        return fun(*args)

    x = args[argnum]
    eps = 1e-4
    if isinstance(x, np.ndarray):
        nd_grad = np.zeros(x.shape)
        for dims in it.product(*map(range, x.shape)):
            eps_vector = np.zeros(x.shape)
            eps_vector[dims] = eps
            nd_grad[dims] = (f(x + eps_vector/2) - f(x - eps_vector/2)) / eps
        return nd_grad
    else:
        return (f(x + eps/2) - f(x - eps/2)) / eps

def check_grads(fun, *args):
    for i, arg in enumerate(args):
        gradfun = grad(fun, i)
        A = nd(fun, args, i)
        B = gradfun(*args)
        if isinstance(A, np.ndarray) or isinstance(B, np.ndarray):
            assert A.shape == B.shape, \
                "In arg num {0}, with args {1}, shapes {2} and {3}".format(
                    i, args, A.shape, B.shape)
        assert np.allclose(A, B, rtol=1e-4, atol=1e-6), \
            "In arg num {0}, with args {1}, difs are: {2}".format(
                i, args, A - B)

def to_scalar(x):
    return np.sum(np.sin(x))
