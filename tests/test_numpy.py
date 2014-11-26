import numpy as np
import numpy.random as npr
from test_util import *
from funkyyak import grad, kyapply
k = kyapply
npr.seed(1)

def test_dot():
    def fun(x, y): return to_scalar(k(np.dot, x, y))

    df_0 = grad(fun, argnum=0)
    df_1 = grad(fun, argnum=1)

    mat1 = npr.randn(10, 11)
    mat2 = npr.randn(10, 11)
    vect1 = npr.randn(10)
    vect2 = npr.randn(11)
    vect3 = npr.randn(11)

    check_grads(fun, mat1, vect2)
    check_grads(fun, mat1, mat2.T)
    check_grads(fun, vect1, mat1)
    check_grads(fun, vect2, vect3)

# TODO:
# reshape, squeeze, transpose, getitem
