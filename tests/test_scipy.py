import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.misc
import autograd.scipy.signal
import autograd.scipy.stats

from autograd import grad
from numpy_utils import combo_check, unary_ufunc_check, check_grads
npr.seed(1)


def test_norm_pdf2():
    x = npr.randn()
    fun = lambda l : autograd.scipy.stats.norm.pdf(x=npr.randn(), loc=l, scale=npr.rand()**2)
    d_fun = grad(fun, x)
    check_grads(fun, x)
    check_grads(d_fun, x)
test_norm_pdf2()

def test_norm_pdf(): unary_ufunc_check(autograd.scipy.stats.norm.pdf, loc=[1.1], scale=[0.4])
#def test_norm_pdf(): unary_ufunc_check(lambda l: autograd.scipy.stats.norm.pdf(x = 0.1, loc=l, scale = 1.1))
def test_norm_cdf(): unary_ufunc_check(autograd.scipy.stats.norm.cdf, loc=[1.1], scale=[0.4])


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

def test_convolve(): combo_check(autograd.scipy.signal.convolve, [0,1],
                                 [R(4), R(5), R(6)],
                                 [R(2), R(3), R(4)],
                                 mode=['full', 'valid', 'same'])

def test_convolve2d(): combo_check(autograd.scipy.signal.convolve2d, [0, 1],
                                   [R(4, 3), R(5, 4), R(6, 7)],
                                   [R(2, 2), R(3, 2), R(4, 3)],
                                   mode=['full', 'valid', 'same'])

