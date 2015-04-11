import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.util import *
from autograd import grad
npr.seed(1)

def test_fft():
    def fun(x): return to_scalar(np.fft.fft(x))
    d_fun = lambda x : to_scalar(grad(fun)(x))
    D = 5
    mat = npr.randn(D, D)
    check_grads(fun, mat)
    check_grads(d_fun, mat)

def test_fft_axis():
    def fun(x): return to_scalar(np.fft.fft(x, axis=0))
    d_fun = lambda x : to_scalar(grad(fun)(x))
    D = 5
    mat = npr.randn(D, D)
    check_grads(fun, mat)
    check_grads(d_fun, mat)

#def test_fft_n_smaller():
#    D = 5
#    def fun(x): return to_scalar(np.fft.fft(x, D - 2))
#    d_fun = lambda x : to_scalar(grad(fun)(x))
#    mat = npr.randn(D, D)
#    check_grads(fun, mat)
#    check_grads(d_fun, mat)

#def test_fft_n_bigger():
#    D = 5
#    def fun(x): return to_scalar(np.fft.fft(x, D + 2))
#    d_fun = lambda x : to_scalar(grad(fun)(x))
#    mat = npr.randn(D, D)
#    check_grads(fun, mat)
#    check_grads(d_fun, mat)

def test_ifft():
    def fun(x): return to_scalar(np.fft.ifft(x))
    d_fun = lambda x : to_scalar(grad(fun)(x))
    D = 5
    mat = npr.randn(D, D)
    check_grads(fun, mat)
    check_grads(d_fun, mat)

def test_fft2():
    def fun(x): return to_scalar(np.fft.fft2(x))
    d_fun = lambda x : to_scalar(grad(fun)(x))
    D = 5
    mat = npr.randn(D, D) / 10.0
    check_grads(fun, mat)
    check_grads(d_fun, mat)

def test_ifft2():
    def fun(x): return to_scalar(np.fft.ifft2(x))
    d_fun = lambda x : to_scalar(grad(fun)(x))
    D = 5
    mat = npr.randn(D, D)
    check_grads(fun, mat)
    check_grads(d_fun, mat)

def test_fftn():
    def fun(x): return to_scalar(np.fft.fftn(x))
    d_fun = lambda x : to_scalar(grad(fun)(x))
    D = 5
    mat = npr.randn(D, D) / 10.0
    check_grads(fun, mat)
    check_grads(d_fun, mat)

def test_ifftn():
    def fun(x): return to_scalar(np.fft.ifftn(x))
    d_fun = lambda x : to_scalar(grad(fun)(x))
    D = 5
    mat = npr.randn(D, D)
    check_grads(fun, mat)
    check_grads(d_fun, mat)