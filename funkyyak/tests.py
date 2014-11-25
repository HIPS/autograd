import numpy as np
import numpy.random as npr
import operator as op
import itertools as it
from funkyyak import grad, kyapply

k = kyapply

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
        num_grad = nd(fun, args, i)
        an_grad = gradfun(*args)
        assert_close(an_grad, num_grad)

def assert_close(A, B):
    assert A.shape == B.shape
    assert np.allclose(A, B, rtol=1e-4, atol=1e-6)

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
    randv = npr.randn(10)
    def fun(x):
        return k(np.sin, k(np.dot, x, randv))
    df = grad(fun)
    def vector_product(x, v):
        return k(np.sin, k(np.dot, v, df(x)))
    ddf = grad(vector_product)
    A = npr.randn(10)
    B = npr.randn(10)
    assert_close(df(A), nd(fun, (A,), 0))
    assert_close(ddf(A, B), nd(vector_product, (A, B), 0))

def test_dot():
    npr.seed(1)
    def fun(x, y):
        return k(np.sum, k(np.sin, k(np.dot, x, y)))
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

def test_grad_mul():
    npr.seed(2)
    def fun(x, y):
        return k(np.sum, k(np.sin, x * y))

    scalar = 5.0
    vector = npr.randn(6)
    mat = npr.randn(12, 6)
    mat2 = npr.randn(1, 6)
    allargs = [scalar, vector, mat, mat2]

    for arg1, arg2 in it.product(allargs, allargs):
        check_grads(fun, arg1, arg2)
