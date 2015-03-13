import numpy as np
import numpy.random as npr
import scipy.stats as sps
from test_util import *
from autograd import grad
npr.seed(1)

def test_abs():
    fun = lambda x : 3.0 * np.abs(x)
    d_fun = grad(fun)
    check_grads(fun, 1.1)
    check_grads(fun, -1.1)
    check_grads(d_fun, 1.1)
    check_grads(d_fun, -1.1)

def test_sin():
    fun = lambda x : 3.0 * np.sin(x)
    d_fun = grad(fun)
    check_grads(fun, npr.randn())
    check_grads(d_fun, npr.randn())

def test_sign():
    fun = lambda x : 3.0 * np.sign(x)
    d_fun = grad(fun)
    check_grads(fun, 1.1)
    check_grads(fun, -1.1)
    check_grads(d_fun, 1.1)
    check_grads(d_fun, -1.1)

def test_exp():
    fun = lambda x : 3.0 * np.exp(x)
    d_fun = grad(fun)
    check_grads(fun, npr.randn())
    check_grads(d_fun, npr.randn())

def test_log():
    fun = lambda x : 3.0 * np.log(x)
    d_fun = grad(fun)
    check_grads(fun, abs(npr.randn()))
    check_grads(d_fun, abs(npr.randn()))

def test_neg():
    fun = lambda x : 3.0 * - x
    d_fun = grad(fun)
    check_grads(fun, npr.randn())
    check_grads(d_fun, npr.randn())

def test_cos():
    fun = lambda x : 3.0 * np.cos(x)
    d_fun = grad(fun)
    check_grads(fun, npr.randn())
    check_grads(d_fun, npr.randn())

def test_tan():
    fun = lambda x : 3.0 * np.tan(x)
    d_fun = grad(fun)
    check_grads(fun, npr.randn())
    check_grads(d_fun, npr.randn())

def test_cosh():
    fun = lambda x : 3.0 * np.cosh(x)
    d_fun = grad(fun)
    check_grads(fun, npr.randn())
    check_grads(d_fun, npr.randn())

def test_sinh():
    fun = lambda x : 3.0 * np.sinh(x)
    d_fun = grad(fun)
    check_grads(fun, npr.randn())
    check_grads(d_fun, npr.randn())

def test_tanh():
    fun = lambda x : 3.0 * np.tanh(x)
    d_fun = grad(fun)
    check_grads(fun, npr.randn())
    check_grads(d_fun, npr.randn())

def test_sqrt():
    fun = lambda x : 3.0 * np.sqrt(x)
    d_fun = grad(fun)
    check_grads(fun, 10.0*npr.rand())
    check_grads(d_fun, 10.0*npr.rand())

def norm_cdf():
    loc = npr.randn()
    fun = lambda x,loc,scale : 3.0 * sps.norm.cdf(x, loc=loc, scale=scale)
    d_fun = grad(fun)
    check_grads(fun, npr.randn(), loc=npr.randn(), scale=npr.rand())
    check_grads(d_fun, npr.randn(), loc=npr.randn(), scale=npr.rand())
