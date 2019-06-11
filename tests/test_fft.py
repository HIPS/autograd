from __future__ import absolute_import
from functools import partial
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.test_util import check_grads
from autograd import grad
import pytest
npr.seed(1)

### fwd mode not yet implemented
check_grads = partial(check_grads, modes=['rev'])

def test_fft():
    def fun(x): return np.fft.fft(x)
    D = 5
    mat = npr.randn(D, D)
    check_grads(fun)(mat)

def test_fft_ortho():
    def fun(x): return np.fft.fft(x, norm='ortho')
    D = 5
    mat = npr.randn(D, D)
    check_grads(fun)(mat)

def test_fft_axis():
    def fun(x): return np.fft.fft(x, axis=0)
    D = 5
    mat = npr.randn(D, D)
    check_grads(fun)(mat)

def match_complex(fft_fun, mat):
    # ensure hermitian by doing a fft
    if fft_fun.__name__.startswith('ir'):
        return getattr(np.fft, fft_fun.__name__[1:])(mat)
    else:
        return mat

def check_fft_n(fft_fun, D, n):
    def fun(x): return fft_fun(x, D + n)
    mat = npr.randn(D, D)
    mat = match_complex(fft_fun, mat)
    check_grads(fun)(mat)

def test_fft_n_smaller(): check_fft_n(np.fft.fft, 5, -2)
def test_fft_n_bigger(): check_fft_n(np.fft.fft, 5, 2)
def test_ifft_n_smaller(): check_fft_n(np.fft.ifft, 5, -2)
def test_ifft_n_bigger(): check_fft_n(np.fft.ifft, 5, 2)

def test_rfft_n_smaller(): check_fft_n(np.fft.rfft, 4, -2)
def test_rfft_n_bigger(): check_fft_n(np.fft.rfft, 4, 2)
def test_irfft_n_smaller(): check_fft_n(np.fft.irfft, 4, -2)
def test_irfft_n_bigger(): check_fft_n(np.fft.irfft, 4, 2)

def check_fft_s(fft_fun, D):
   def fun(x): return fft_fun(x, s=s, axes=axes)
   mat = npr.randn(D,D,D) / 10.0
   mat = match_complex(fft_fun, mat)
   s = [D + 2, D - 2]
   axes = [0,2]
   check_grads(fun)(mat)

def test_fft2_s():  check_fft_s(np.fft.fft2, 5)
def test_ifft2_s(): check_fft_s(np.fft.ifft2, 5)
def test_fftn_s():  check_fft_s(np.fft.fftn, 5)
def test_ifftn_s(): check_fft_s(np.fft.ifftn, 5)

def test_rfft2_s():  check_fft_s(np.fft.rfft2, 4)
def test_irfft2_s(): check_fft_s(np.fft.irfft2, 4)
def test_rfftn_s():  check_fft_s(np.fft.rfftn, 4)
def test_irfftn_s(): check_fft_s(np.fft.irfftn, 4)

## TODO: fft gradient not implemented for repeated axes
# def test_fft_repeated_axis():
#     D = 5
#     for fft_fun in (np.fft.fft2,np.fft.ifft2,np.fft.fftn, np.fft.ifftn):
#        def fun(x): return fft_fun(x, s=s, axes=axes)

#        mat = npr.randn(D,D,D) / 10.0
#        s = [D + 2, D - 2]
#        axes = [0,0]

#   check_grads(rad)(fun)

def test_ifft():
    def fun(x): return np.fft.ifft(x)
    D = 5
    mat = npr.randn(D, D)
    check_grads(fun)(mat)

def test_fft2():
    def fun(x): return np.fft.fft2(x)
    D = 5
    mat = npr.randn(D, D) / 10.0
    check_grads(fun)(mat)

def test_ifft2():
    def fun(x): return np.fft.ifft2(x)
    D = 5
    mat = npr.randn(D, D)
    check_grads(fun)(mat)

def test_fftn():
    def fun(x): return np.fft.fftn(x)
    D = 5
    mat = npr.randn(D, D) / 10.0
    check_grads(fun)(mat)

def test_ifftn():
    def fun(x): return np.fft.ifftn(x)
    D = 5
    mat = npr.randn(D, D)
    check_grads(fun)(mat)

def test_rfft():
    def fun(x): return np.fft.rfft(x)
    D = 4
    mat = npr.randn(D, D) / 10.0
    check_grads(fun)(mat)

def test_rfft_ortho():
    def fun(x): return np.fft.rfft(x, norm='ortho')
    D = 4
    mat = npr.randn(D, D) / 10.0
    check_grads(fun)(mat)

def test_rfft_axes():
    def fun(x): return np.fft.rfft(x, axis=0)
    D = 4
    mat = npr.randn(D, D) / 10.0
    check_grads(fun)(mat)

def test_irfft():
    def fun(x): return np.fft.irfft(x)
    D = 4
    mat = npr.randn(D, D) / 10.0
    # ensure hermitian by doing a fft
    mat = np.fft.rfft(mat)
    check_grads(fun)(mat)

def test_irfft_ortho():
    def fun(x): return np.fft.irfft(x, norm='ortho')
    D = 4
    mat = npr.randn(D, D) / 10.0
    # ensure hermitian by doing a fft
    mat = np.fft.rfft(mat)
    check_grads(fun)(mat)

def test_rfft2():
    def fun(x): return np.fft.rfft2(x)
    D = 4
    mat = npr.randn(D, D) / 10.0
    check_grads(fun)(mat)

def test_irfft2():
    def fun(x): return np.fft.irfft2(x)
    D = 4
    mat = npr.randn(D, D) / 10.0
    # ensure hermitian by doing a fft
    mat = np.fft.rfft2(mat)
    check_grads(fun)(mat)

def test_rfftn():
    def fun(x): return np.fft.rfftn(x)
    D = 4
    mat = npr.randn(D, D, D) / 10.0
    check_grads(fun)(mat)

def test_rfftn_odd_not_implemented():
    def fun(x): return np.fft.rfftn(x)
    D = 5
    mat = npr.randn(D, D, D) / 10.0
    with pytest.raises(NotImplementedError):
        check_grads(fun)(mat)

def test_rfftn_subset():
    def fun(x): return np.fft.rfftn(x)[(0, 1, 0), (3, 3, 2)]
    D = 4
    mat = npr.randn(D, D, D) / 10.0
    check_grads(fun)(mat)

def test_rfftn_axes():
    def fun(x): return np.fft.rfftn(x, axes=(0, 2))
    D = 4
    mat = npr.randn(D, D, D) / 10.0
    check_grads(fun)(mat)

def test_irfftn():
    def fun(x): return np.fft.irfftn(x)
    D = 4
    mat = npr.randn(D, D, D) / 10.0
    # ensure hermitian by doing a fft
    mat = np.fft.rfftn(mat)
    check_grads(fun)(mat)

def test_irfftn_subset():
    def fun(x): return np.fft.irfftn(x)[(0, 1, 0), (3, 3, 2)]
    D = 4
    mat = npr.randn(D, D, D) / 10.0
    # ensure hermitian by doing a fft
    mat = np.fft.rfftn(mat)
    check_grads(fun)(mat)

def test_fftshift():
    def fun(x): return np.fft.fftshift(x)
    D = 5
    mat = npr.randn(D, D) / 10.0
    check_grads(fun)(mat)

def test_fftshift_even():
    def fun(x): return np.fft.fftshift(x)
    D = 4
    mat = npr.randn(D, D) / 10.0
    check_grads(fun)(mat)

def test_fftshift_axes():
    def fun(x): return np.fft.fftshift(x, axes=1)
    D = 5
    mat = npr.randn(D, D) / 10.0
    check_grads(fun)(mat)

def test_ifftshift():
    def fun(x): return np.fft.ifftshift(x)
    D = 5
    mat = npr.randn(D, D)
    check_grads(fun)(mat)

def test_ifftshift_even():
    def fun(x): return np.fft.ifftshift(x)
    D = 4
    mat = npr.randn(D, D)
    check_grads(fun)(mat)

def test_ifftshift_axes():
    def fun(x): return np.fft.ifftshift(x, axes=1)
    D = 5
    mat = npr.randn(D, D)
    check_grads(fun)(mat)
