import numpy as np
import numpy.random as npr
import operator as op
import itertools as it
from core import grad, kyapply

k = kyapply

def nd(f, x):
    eps = 1e-4
    nd_grad = np.zeros(x.shape)
    for dims in it.product(*map(range, x.shape)):
        eps_vector = np.zeros(x.shape)
        eps_vector[dims] = eps
        nd_grad[dims] = (f(x + eps_vector/2) - f(x - eps_vector/2)) / eps
    return nd_grad

def test_grad_pow():
    fun = lambda x, y : k(op.pow, x, y)
    df = grad(fun)
    assert np.allclose(df(3.0, 4), 108.0)

def test_grad_sin():
    fun = lambda x : k(np.sin, x)
    df = grad(fun)
    assert np.allclose(df(np.pi/3), 0.5)

def test_grad_fanout():
    fun = lambda x : k(np.sin, x) + k(np.sin, x)
    df = grad(fun)
    assert np.allclose(df(np.pi/3), 1.0)

def test_grad_const():
    fun = lambda x : 1
    df = grad(fun)
    assert np.allclose(df(2.0), 0.0)

def test_grad_exp():
    fun = lambda x : k(np.exp, x)
    df = grad(fun)
    assert np.allclose(df(2.0), np.exp(2.0))

def test_double_grad_exp():
    fun = lambda x : k(np.exp, x)
    df = grad(fun)
    ddf = grad(grad(df))
    assert np.allclose(df(2.0), np.exp(2.0))
    assert np.allclose(ddf(2.0), np.exp(2.0))

def test_grad_identity():
    fun = lambda x : x
    df = grad(fun)
    ddf = grad(df)
    assert np.allclose(df(2.0), 1.0)
    assert np.allclose(ddf(2.0), 0.0)

def test_double_grad_sin():
    fun = lambda x : k(np.sin, x)
    ddf = grad(grad(fun))
    assert np.allclose(ddf(np.pi/6), -0.5)

def test_hess_vector_prod():
    npr.seed(1)
    randv = npr.randn(20)
    def fun(x):
        return k(np.sin, k(np.dot, x, randv))
    df = grad(fun)
    def vector_product(x, v):
        return k(np.sin, k(np.dot, v, df(x)))
    ddf = grad(vector_product)
    A = npr.randn(20)
    B = npr.randn(20)
    assert np.allclose(df(A), nd(fun, A))
    assert np.allclose(ddf(A, B), nd(lambda x : vector_product(x, B), A))
