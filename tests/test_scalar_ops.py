import numpy as np
import numpy.random as npr
from test_util import *
from funkyyak import grad, kyapply
k = kyapply
npr.seed(1)

def test_abs():
    fun = lambda x : k(np.abs, x)
    d_fun = grad(fun)
    check_grads(fun, 1.1)
    check_grads(fun, -1.1)
    check_grads(d_fun, 1.1)
    check_grads(d_fun, -1.1)

def test_sin():
    fun = lambda x : np.sin(x)
    d_fun = grad(fun)
    check_grads(fun, npr.randn())
    check_grads(d_fun, npr.randn())

def test_sign():
    fun = lambda x : k(np.sign, x)
    d_fun = grad(fun)
    check_grads(fun, 1.1)
    check_grads(fun, -1.1)
    check_grads(d_fun, 1.1)
    check_grads(d_fun, -1.1)

def test_exp():
    fun = lambda x : np.exp(x)
    d_fun = grad(fun)
    check_grads(fun, npr.randn())
    check_grads(d_fun, npr.randn())

def test_log():
    fun = lambda x : np.log(x)
    d_fun = grad(fun)
    print npr.randn()
    print npr.randn()
    check_grads(fun, abs(npr.randn()))
    check_grads(d_fun, abs(npr.randn()))

def test_neg():
    fun = lambda x : - x
    d_fun = grad(fun)
    check_grads(fun, npr.randn())
    check_grads(d_fun, npr.randn())
