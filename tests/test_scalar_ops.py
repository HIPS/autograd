import autograd.numpy as np
import autograd.numpy.random as npr
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

def test_arccosh():
    fun = lambda x : 3.0 * np.arccosh(x)
    d_fun = grad(fun)
    check_grads(fun, npr.randn()**2 + 1)
    check_grads(d_fun, npr.randn()**2 + 1)

def test_arcsinh():
    fun = lambda x : 3.0 * np.arcsinh(x)
    d_fun = grad(fun)
    check_grads(fun, npr.randn())
    check_grads(d_fun, npr.randn())

def test_arctanh():
    fun = lambda x : 3.0 * np.arctanh(x)
    d_fun = grad(fun)
    check_grads(fun, 0.2)
    check_grads(d_fun, 0.3)

def test_sqrt():
    fun = lambda x : 3.0 * np.sqrt(x)
    d_fun = grad(fun)
    check_grads(fun, 10.0*npr.rand())
    check_grads(d_fun, 10.0*npr.rand())

def test_power_arg0():
    y = npr.randn()**2
    fun = lambda x : np.power(x, y)
    d_fun = grad(fun)
    check_grads(fun, npr.rand()**2)
    check_grads(d_fun, npr.rand()**2)

def test_power_arg1():
    x = npr.randn()**2
    fun = lambda y : np.power(x, y)
    d_fun = grad(fun)
    check_grads(fun, npr.rand()**2)
    check_grads(d_fun, npr.rand()**2)

def test_mod_arg0():
    fun = lambda x, y : np.mod(x, y)
    d_fun = grad(fun)
    check_grads(fun, npr.rand(), npr.rand())
    check_grads(d_fun, npr.rand(), npr.rand())

def test_mod_arg1():
    fun = lambda x, y : np.mod(x, y)
    d_fun = grad(fun, 1)
    check_grads(fun, npr.rand(), npr.rand())
    check_grads(d_fun, npr.rand(), npr.rand())


#def test_norm_cdf():
#    x = npr.randn()
#    fun = lambda x : 3.0 * sps.norm.cdf(x, loc=npr.randn(), scale=npr.rand()**2)
#    d_fun = grad(fun, x)
#    check_grads(fun, x)
#    check_grads(d_fun, x)
