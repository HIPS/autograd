from __future__ import absolute_import
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.util import *
from autograd.container_types import make_tuple
from autograd import grad
npr.seed(1)

def test_getter():
    def fun(input_tuple):
        A = np.sum(input_tuple[0])
        B = np.sum(input_tuple[1])
        C = np.sum(input_tuple[1])
        return A + B + C

    d_fun = grad(fun)
    input_tuple = (npr.randn(5, 6),
                   npr.randn(4, 3),
                   npr.randn(2, 4))

    result = d_fun(input_tuple)
    print(result)
    assert np.allclose(result[0], np.ones((5, 6)))
    assert np.allclose(result[1], 2 * np.ones((4, 3)))
    assert np.allclose(result[2], np.zeros((2, 4)))

def test_grads():
    def fun(input_tuple):
        A = np.sum(np.sin(input_tuple[0]))
        B = np.sum(np.cos(input_tuple[1]))
        return A + B

    def d_fun(input_tuple):
        g = grad(fun)(input_tuple)
        A = np.sum(g[0])
        B = np.sum(np.sin(g[0]))
        C = np.sum(np.sin(g[1]))
        return A + B + C

    input_tuple = (npr.randn(5, 6),
                   npr.randn(4, 3),
                   npr.randn(2, 4))

    check_grads(fun, input_tuple)
    check_grads(d_fun, input_tuple)

def test_nested_higher_order():
    def outer_fun(x):
        def inner_fun(y):
            return y[0] * y[1]
        return np.sum(np.sin(np.array(grad(inner_fun)(make_tuple(x,x)))))

    check_grads(outer_fun, 5.)
    check_grads(grad(outer_fun), 10.)
    check_grads(grad(grad(outer_fun)), 10.)
