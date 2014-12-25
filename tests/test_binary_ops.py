import numpy as np
import numpy.random as npr
from test_util import *
from funkyyak import grad
npr.seed(1)

def arg_pairs():
    scalar = 2.0
    vector = npr.randn(6)
    mat = npr.randn(7, 6)
    mat2 = npr.randn(1, 6)
    allargs = [scalar, vector, mat, mat2]
    for arg1, arg2 in it.product(allargs, allargs):
        yield arg1, arg2

def test_mul():
    fun = lambda x, y : to_scalar(x * y)
    d_fun_0 = lambda x, y : to_scalar(grad(fun, 0)(x, y))
    d_fun_1 = lambda x, y : to_scalar(grad(fun, 1)(x, y))
    for arg1, arg2 in arg_pairs():
        check_grads(fun, arg1, arg2)
        check_grads(d_fun_0, arg1, arg2)
        check_grads(d_fun_1, arg1, arg2)

def test_add():
    fun = lambda x, y : to_scalar(x + y)
    d_fun_0 = lambda x, y : to_scalar(grad(fun, 0)(x, y))
    d_fun_1 = lambda x, y : to_scalar(grad(fun, 1)(x, y))
    for arg1, arg2 in arg_pairs():
        check_grads(fun, arg1, arg2)
        check_grads(d_fun_0, arg1, arg2)
        check_grads(d_fun_1, arg1, arg2)

def test_sub():
    fun = lambda x, y : to_scalar(x - y)
    d_fun_0 = lambda x, y : to_scalar(grad(fun, 0)(x, y))
    d_fun_1 = lambda x, y : to_scalar(grad(fun, 1)(x, y))
    for arg1, arg2 in arg_pairs():
        check_grads(fun, arg1, arg2)
        check_grads(d_fun_0, arg1, arg2)
        check_grads(d_fun_1, arg1, arg2)

def test_div():
    fun = lambda x, y : to_scalar(x / y)
    d_fun_0 = lambda x, y : to_scalar(grad(fun, 0)(x, y))
    d_fun_1 = lambda x, y : to_scalar(grad(fun, 1)(x, y))
    make_gap_from_zero = lambda x : np.sqrt(x **2 + 0.5)
    for arg1, arg2 in arg_pairs():
        arg1 = make_gap_from_zero(arg1)
        arg2 = make_gap_from_zero(arg2)
        check_grads(fun, arg1, arg2)
        check_grads(d_fun_0, arg1, arg2)
        check_grads(d_fun_1, arg1, arg2)

def test_pow():
    fun = lambda x, y : to_scalar(x ** y)
    d_fun_0 = lambda x, y : to_scalar(grad(fun, 0)(x, y))
    d_fun_1 = lambda x, y : to_scalar(grad(fun, 1)(x, y))
    make_positive = lambda x : np.abs(x) + 1.1 # Numeric derivatives fail near zero
    for arg1, arg2 in arg_pairs():
        arg1 = make_positive(arg1)
        arg2 = np.round(arg2)
        check_grads(fun, arg1, arg2)
        check_grads(d_fun_0, arg1, arg2)
        check_grads(d_fun_1, arg1, arg2)
