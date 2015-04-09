from __future__ import absolute_import
from scipy.signal import convolve as sp_convolve
import numpy as npo
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.misc
import autograd.scipy.signal
import autograd.scipy.stats

from autograd import grad

from numpy_utils import combo_check, check_grads, unary_ufunc_check, to_scalar
npr.seed(1)

### Stats ###

def test_norm_pdf():
    x = npr.randn()
    l = npr.randn()
    scale=npr.rand()**2 + 1.1
    fun = autograd.scipy.stats.norm.pdf
    d_fun = grad(fun)
    check_grads(fun, x, l, scale)
    check_grads(d_fun, x, l, scale)

def test_norm_cdf():
    x = npr.randn()
    l = npr.randn()
    scale=npr.rand()**2 + 1.1
    fun = autograd.scipy.stats.norm.cdf
    d_fun = grad(fun)
    check_grads(fun, x, l, scale)
    check_grads(d_fun, x, l, scale)

def test_norm_logpdf():
    x = npr.randn()
    l = npr.randn()
    scale=npr.rand()**2 + 1.1
    fun = autograd.scipy.stats.norm.logpdf
    d_fun = grad(fun)
    check_grads(fun, x, l, scale)
    check_grads(d_fun, x, l, scale)


### Misc ###

R = npr.randn
def test_logsumexp1(): combo_check(autograd.scipy.misc.logsumexp, [0],
                                   [R(4), R(3,4)], axis=[None, 0])
def test_logsumexp2(): combo_check(autograd.scipy.misc.logsumexp, [0],
                                   [R(3,4), R(4,5,6)], axis=[None, 0, 1])
def test_logsumexp3(): combo_check(autograd.scipy.misc.logsumexp, [0],
                                   [R(4)], b = [np.exp(R(4))], axis=[None, 0])
def test_logsumexp4(): combo_check(autograd.scipy.misc.logsumexp, [0],
                                   [R(3,4),], b = [np.exp(R(3,4))], axis=[None, 0, 1])
def test_logsumexp5(): combo_check(autograd.scipy.misc.logsumexp, [0],
                                   [R(2,3,4)], b = [np.exp(R(2,3,4))], axis=[None, 0, 1])

### Signal ###

def test_convolve_generalization():
    ag_convolve = autograd.scipy.signal.convolve
    A_35 = R(3, 5)
    A_34 = R(3, 4)
    A_342 = R(3, 4, 2)
    A_342 = R(3, 4, 2)
    A_2543 = R(2, 5, 4, 3)
    A_24232 = R(2, 4, 2, 3, 2)

    for mode in ['valid', 'full']:
        assert npo.allclose(ag_convolve(A_35,      A_34, axes=([1], [0]), mode=mode)[1, 2],
                            sp_convolve(A_35[1,:], A_34[:, 2], mode))
        assert npo.allclose(ag_convolve(A_35, A_34, axes=([],[]), dot_axes=([0], [0]), mode=mode),
                            npo.tensordot(A_35, A_34, axes=([0], [0])))
        assert npo.allclose(ag_convolve(A_35, A_342, axes=([1],[2]),
                                        dot_axes=([0], [0]), mode=mode)[2],
                            sum([sp_convolve(A_35[i, :], A_342[i, 2, :], mode)
                                 for i in range(3)]))
        assert npo.allclose(ag_convolve(A_2543, A_24232, axes=([1, 2],[2, 4]),
                                        dot_axes=([0, 3], [0, 3]), mode=mode)[2],
                            sum([sum([sp_convolve(A_2543[i, :, :, j],
                                                 A_24232[i, 2, :, j, :], mode)
                                      for i in range(2)]) for j in range(3)]))

def test_convolve(): combo_check(autograd.scipy.signal.convolve, [0,1],
                                 [R(4), R(5), R(6)],
                                 [R(2), R(3), R(4)],
                                 mode=['full', 'valid'])

def test_convolve_2d(): combo_check(autograd.scipy.signal.convolve, [0, 1],
                                    [R(4, 3), R(5, 4), R(6, 7)],
                                    [R(2, 2), R(3, 2), R(4, 2), R(4, 1)],
                                   mode=['full', 'valid'])

def test_convolve_ignore(): combo_check(autograd.scipy.signal.convolve, [0, 1],
                                        [R(4, 3)], [R(3, 2)],
                                        axes=[([0],[0]), ([1],[1]), ([0],[1]), ([1],[0]),
                                              ([0, 1], [0, 1]), ([1, 0], [1, 0])],
                                        mode=['full', 'valid'])

def test_convolve_ignore_dot(): combo_check(autograd.scipy.signal.convolve, [0, 1],
                                            [R(3, 3, 2)], [R(3, 2, 3)],
                                            axes=[([1],[1])],
                                            dot_axes=[([0],[2]), ([0],[0])],
                                            mode=['full', 'valid'])
### Special ###

def test_polygamma():
    x = npr.randn()
    fun = lambda x: to_scalar(autograd.scipy.special.polygamma(0, x))
    d_fun = grad(fun)
    check_grads(fun, x)
    check_grads(d_fun, x)

def test_psi():     unary_ufunc_check(autograd.scipy.special.psi,     lims=[0.3, 2.0])
def test_digamma(): unary_ufunc_check(autograd.scipy.special.digamma, lims=[0.3, 2.0])
def test_gamma():   unary_ufunc_check(autograd.scipy.special.gamma,   lims=[0.3, 2.0])

def test_j0(): unary_ufunc_check(autograd.scipy.special.j0, lims=[0.2, 20.0])
def test_j1(): unary_ufunc_check(autograd.scipy.special.j1, lims=[0.2, 20.0])
def test_y0(): unary_ufunc_check(autograd.scipy.special.y0, lims=[0.2, 20.0])
def test_y1(): unary_ufunc_check(autograd.scipy.special.y1, lims=[0.2, 20.0])

def test_jn():
    x = npr.randn()**2 + 0.3
    fun = lambda x: to_scalar(autograd.scipy.special.jn(2, x))
    d_fun = grad(fun)
    check_grads(fun, x)
    check_grads(d_fun, x)

def test_yn():
    x = npr.randn()**2 + 0.3
    fun = lambda x: to_scalar(autograd.scipy.special.yn(2, x))
    d_fun = grad(fun)
    check_grads(fun, x)
    check_grads(d_fun, x)
