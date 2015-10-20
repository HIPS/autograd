from __future__ import absolute_import
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.util import *
from autograd import grad
npr.seed(1)

def test_inv():
    def fun(x): return to_scalar(np.linalg.inv(x))
    d_fun = lambda x : to_scalar(grad(fun)(x))
    D = 8
    mat = npr.randn(D, D)
    mat = np.dot(mat, mat) + 1.0 * np.eye(D)
    check_grads(fun, mat)
    check_grads(d_fun, mat)

def test_solve_arg1():
    D = 8
    A = npr.randn(D, D) + 10.0 * np.eye(D)
    B = npr.randn(D, D - 1)
    def fun(a): return to_scalar(np.linalg.solve(a, B))
    d_fun = lambda x : to_scalar(grad(fun)(x))
    check_grads(fun, A)
    check_grads(d_fun, A)

def test_solve_arg1_1d():
    D = 8
    A = npr.randn(D, D) + 10.0 * np.eye(D)
    B = npr.randn(D)
    def fun(a): return to_scalar(np.linalg.solve(a, B))
    d_fun = lambda x : to_scalar(grad(fun)(x))
    check_grads(fun, A)
    check_grads(d_fun, A)

#return np.dot(b,np.dot(np.linalg.inv(A),b))

def test_solve_arg2():
    D = 6
    A = npr.randn(D, D) + 1.0 * np.eye(D)
    B = npr.randn(D, D - 1)
    def fun(b): return to_scalar(np.linalg.solve(A, b))
    d_fun = lambda x : to_scalar(grad(fun)(x))
    check_grads(fun, B)
    check_grads(d_fun, B)

def test_det():
    def fun(x): return np.linalg.det(x)
    d_fun = lambda x : to_scalar(grad(fun)(x))
    D = 6
    mat = npr.randn(D, D)
    check_grads(fun, mat)
    check_grads(d_fun, mat)

def test_slogdet():
    def fun(x):
        sign, logdet = np.linalg.slogdet(x)
        return logdet
    d_fun = lambda x : to_scalar(grad(fun)(x))
    D = 6
    mat = npr.randn(D, D)
    check_grads(fun, mat)
    check_grads(d_fun, mat)

def test_vector_2norm():
    def fun(x): return to_scalar(np.linalg.norm(x))
    d_fun = lambda x: to_scalar(grad(fun)(x))
    D = 6
    vec = npr.randn(D)
    check_grads(fun, vec)
    check_grads(d_fun, vec)

def test_frobenius_norm():
    def fun(x): return to_scalar(np.linalg.norm(x))
    d_fun = lambda x : to_scalar(grad(fun)(x))
    D = 6
    mat = npr.randn(D, D-1)
    check_grads(fun, mat)
    check_grads(d_fun, mat)

def test_vector_norm_ord():
    def helper(size, ord):
        def fun(x): return np.linalg.norm(x, ord=ord)
        vec = npr.randn(size)
        check_grads(fun, vec)
    for ord in xrange(2,5):
        yield helper, 6, ord

def test_norm_axis():
    def helper(shape, axis):
        def fun(x): return to_scalar(np.linalg.norm(x, axis=axis))
        arr = npr.randn(*shape)
        check_grads(fun, arr)
    for axis in xrange(3):
        yield helper, (6,5,4), axis

def test_eigvalh_lower():
    def fun(x):
        w, v = np.linalg.eigh(x)
        return to_scalar(w) + to_scalar(v)
    d_fun = lambda x : to_scalar(grad(fun)(x))
    D = 6
    mat = npr.randn(D, D-1)
    hmat = np.dot(mat.T, mat)
    check_grads(fun, hmat)
    check_grads(d_fun, hmat)

def test_eigvalh_upper():
    def fun(x):
        w, v = np.linalg.eigh(x, 'U')
        return to_scalar(w) + to_scalar(v)
    d_fun = lambda x : to_scalar(grad(fun)(x))
    D = 6
    mat = npr.randn(D, D-1)
    hmat = np.dot(mat.T, mat)
    check_grads(fun, hmat)
    check_grads(d_fun, hmat)

def test_cholesky():
    def fun(A):
        return to_scalar(np.linalg.cholesky(A))
    d_fun = lambda A: to_scalar(grad(fun)(A))
    def rand_psd(D):
        mat = npr.randn(D,D)
        return np.dot(mat, mat.T)
    check_grads(fun, rand_psd(6))
    # check_grads(d_fun, rand_psd(6))
