import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.util import *
from autograd import grad
npr.seed(1)

def test_return_both():
    fun = lambda x : 3.0 * np.sin(x)
    d_fun = grad(fun)
    f_and_d_fun = grad(fun, return_function_value=True)

    test_x = npr.randn()
    f, d = f_and_d_fun(test_x)
    assert f == fun(test_x)
    assert d == d_fun(test_x)
