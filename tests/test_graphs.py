from __future__ import absolute_import
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.util import *
from autograd import grad
import warnings
from nose.tools import raises
npr.seed(1)

def test_grad_fanout():
    fun = lambda x : np.sin(np.sin(x) + np.sin(x))
    df = grad(fun)
    check_grads(fun, npr.randn())
    check_grads(df, npr.rand())

def test_grad_const():
    fun = lambda x : 1.0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("ignore")
        df = grad(fun)
        assert np.allclose(df(2.0), 0.0)

def test_grad_identity():
    fun = lambda x : x
    df = grad(fun)
    ddf = grad(df)
    assert np.allclose(df(2.0), 1.0)
    assert np.allclose(ddf(2.0), 0.0)

def test_hess_vector_prod():
    npr.seed(1)
    randv = npr.randn(10)
    def fun(x):
        return np.sin(np.dot(x, randv))
    df = grad(fun)
    def vector_product(x, v):
        return np.sin(np.dot(v, df(x)))
    ddf = grad(vector_product)
    A = npr.randn(10)
    B = npr.randn(10)
    check_grads(fun, A)
    check_grads(vector_product, A, B)

def test_enclosing_scope_ref():
    def fun(x):
        inner_fun = lambda y : x * y
        return x * grad(inner_fun)(2.0)
    check_grads(fun, 1.0)

def test_enclosing_scope_ref_2():
    def fun(x):
        inner_fun = lambda y : y * x
        return x * grad(inner_fun)(2.0)
    check_grads(fun, 1.0)

def test_mutating_outgrad():
    def fun(a):
        b = a + 1.0
        c = b + 1.5
        d = a + b
        e = d + c
        return to_scalar(e)

    A = npr.randn(5)
    check_grads(fun, A)

def test_mutating_outgrad_from_indexing():
    def fun(a):
        b = a + 1.0
        c = b[0] + 1.5
        d = a + b
        e = d + c
        return to_scalar(e)

    A = npr.randn(5)
    check_grads(fun, A)

def test_complex_mutating_outgrad_from_indexing():
    def fun(a):
        b = a + 1.0j
        c = b[0] + 1.5
        d = a + b
        e = d + c
        return to_scalar(e)

    A = npr.randn(5)
    check_grads(fun, A)
    d_fun = lambda x : to_scalar(grad(fun)(x))
    check_grads(d_fun, A)

def test_complex_separate_real_and_imaginary():
    def fun(a):
        r, i = np.real(a), np.imag(a)
        a = np.abs(r)**1.4 + np.abs(i)**1.3
        return to_scalar(a)
    d_fun = lambda x : to_scalar(grad(fun)(x))
    A = npr.randn(5, 3) + 0.1j*npr.randn(5, 3)
    check_grads(fun, A)
    check_grads(d_fun, A)

def test_third_derivative():
    fun = lambda x : np.sin(np.sin(x) + np.sin(x))
    df = grad(fun)
    ddf = grad(fun)
    dddf = grad(fun)
    check_grads(fun, npr.randn())
    check_grads(df, npr.rand())
    check_grads(ddf, npr.rand())
    check_grads(dddf, npr.rand())

def test_third_derivative_other_args():
    fun = lambda x, y : np.sin(np.sin(x) + np.sin(y))
    df = grad(fun)
    ddf = grad(fun, 1)
    dddf = grad(fun)
    check_grads(fun, npr.randn(), npr.randn())
    check_grads(df, npr.randn(), npr.randn())
    check_grads(ddf, npr.randn(), npr.randn())
    check_grads(dddf, npr.randn(), npr.randn())

def test_third_derivative_other_args2():
    fun = lambda x, y : np.sin(np.sin(x) + np.sin(y))
    df = grad(fun, 1)
    ddf = grad(fun)
    dddf = grad(fun, 1)
    check_grads(fun, npr.randn(), npr.randn())
    check_grads(df, npr.randn(), npr.randn())
    check_grads(ddf, npr.randn(), npr.randn())
    check_grads(dddf, npr.randn(), npr.randn())

def test_singleton_array_output():
    fun = lambda x : np.sum(np.sin(x), keepdims=True)
    check_grads(fun, npr.randn(3, 3))
    check_grads(lambda x: np.sum(grad(fun)(x)), npr.randn(3, 3))

def test_singleton_array_output_axis0():
   fun = lambda x : np.sum(np.sin(x), axis=0, keepdims=False)
   check_grads(fun, npr.randn(3, 1))
   check_grads(lambda x: np.sum(grad(fun)(x)), npr.randn(3, 1))

def test_singleton_array_output_axis1():
   fun = lambda x : np.sum(np.sin(x), axis=1, keepdims=False)
   check_grads(fun, npr.randn(1, 3))
   check_grads(lambda x: np.sum(grad(fun)(x)), npr.randn(1, 3))

def test_singleton_array_output_axis0_keepdims():
   fun = lambda x : np.sum(np.sin(x), axis=0, keepdims=True)
   check_grads(fun, npr.randn(3, 1))
   check_grads(lambda x: np.sum(grad(fun)(x)), npr.randn(3, 1))

def test_singleton_array_output_axis1_keepdims():
   fun = lambda x : np.sum(np.sin(x), axis=1, keepdims=True)
   check_grads(fun, npr.randn(1, 3))
   check_grads(lambda x: np.sum(grad(fun)(x)), npr.randn(1, 3))

@raises(TypeError)
def test_assignment_raises_error():
    def fun(A, b):
        A[1] = b
        return to_scalar(A)
    A = npr.randn(5)
    check_grads(fun, A, 3.0)

@raises(TypeError)
def test_nonscalar_output_1():
    grad(lambda x: x * 2)(np.zeros(2))

@raises(TypeError)
def test_nonscalar_output_2():
    grad(lambda x: x * 2)(np.zeros(2))

# TODO:
# Diamond patterns
# Taking grad again after returning const
# Empty functions
# 2nd derivatives with fanout, thinking about the outgrad adder
