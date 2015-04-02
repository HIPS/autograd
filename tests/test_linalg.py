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

def test_frobeneus_norm():
    def fun(x): return to_scalar(np.linalg.norm(x))
    d_fun = lambda x : to_scalar(grad(fun)(x))
    D = 6
    mat = npr.randn(D, D-1)
    check_grads(fun, mat)
    check_grads(d_fun, mat)

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
