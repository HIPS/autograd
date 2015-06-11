from __future__ import absolute_import
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

def test_fft_n_smaller():
   D = 5
   for fft_fun in (np.fft.fft, np.fft.ifft):
       def fun(x): return to_scalar(fft_fun(x, D - 2))
       d_fun = lambda x : to_scalar(grad(fun)(x))
       mat = npr.randn(D, D)
       check_grads(fun, mat)
       check_grads(d_fun, mat)

def test_fft_n_bigger():
   D = 5
   for fft_fun in (np.fft.fft, np.fft.ifft):
       def fun(x): return to_scalar(fft_fun(x, D + 2))
       d_fun = lambda x : to_scalar(grad(fun)(x))
       mat = npr.randn(D, D)
       check_grads(fun, mat)
       check_grads(d_fun, mat)

def check_fft_s(fft_fun):
   D = 5
   def fun(x): return to_scalar(fft_fun(x, s=s, axes=axes))
   d_fun = lambda x : to_scalar(grad(fun)(x))

   mat = npr.randn(D,D,D) / 10.0
   s = [D + 2, D - 2]
   axes = [0,2]

   check_grads(fun, mat)
   check_grads(d_fun, mat)

def test_fft2_s():  check_fft_s(np.fft.fft2)
def test_ifft2_s(): check_fft_s(np.fft.ifft2)
def test_fftn_s():  check_fft_s(np.fft.fftn)
def test_ifftn_s(): check_fft_s(np.fft.ifftn)

## TODO: fft gradient not implemented for repeated axes
# def test_fft_repeated_axis():
#     D = 5
#     for fft_fun in (np.fft.fft2,np.fft.ifft2,np.fft.fftn, np.fft.ifftn):
#        def fun(x): return to_scalar(fft_fun(x, s=s, axes=axes))
#        d_fun = lambda x : to_scalar(grad(fun)(x))

#        mat = npr.randn(D,D,D) / 10.0
#        s = [D + 2, D - 2]
#        axes = [0,0]

#        check_grads(fun, mat)
#        check_grads(d_fun, mat)

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

def test_fftshift():
    def fun(x): return to_scalar(np.fft.fftshift(x))
    d_fun = lambda x : to_scalar(grad(fun)(x))
    D = 5
    mat = npr.randn(D, D) / 10.0
    check_grads(fun, mat)
    check_grads(d_fun, mat)

def test_fftshift_even():
    def fun(x): return to_scalar(np.fft.fftshift(x))
    d_fun = lambda x : to_scalar(grad(fun)(x))
    D = 4
    mat = npr.randn(D, D) / 10.0
    check_grads(fun, mat)
    check_grads(d_fun, mat)

def test_fftshift_axes():
    def fun(x): return to_scalar(np.fft.fftshift(x, axes=1))
    d_fun = lambda x : to_scalar(grad(fun)(x))
    D = 5
    mat = npr.randn(D, D) / 10.0
    check_grads(fun, mat)
    check_grads(d_fun, mat)

def test_ifftshift():
    def fun(x): return to_scalar(np.fft.ifftshift(x))
    d_fun = lambda x : to_scalar(grad(fun)(x))
    D = 5
    mat = npr.randn(D, D)
    check_grads(fun, mat)
    check_grads(d_fun, mat)

def test_ifftshift_even():
    def fun(x): return to_scalar(np.fft.ifftshift(x))
    d_fun = lambda x : to_scalar(grad(fun)(x))
    D = 4
    mat = npr.randn(D, D)
    check_grads(fun, mat)
    check_grads(d_fun, mat)

def test_ifftshift_axes():
    def fun(x): return to_scalar(np.fft.ifftshift(x, axes=1))
    d_fun = lambda x : to_scalar(grad(fun)(x))
    D = 5
    mat = npr.randn(D, D)
    check_grads(fun, mat)
    check_grads(d_fun, mat)
