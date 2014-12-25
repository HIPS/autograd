import numpy as np
import numpy.random as npr
from test_util import *
from funkyyak import grad
npr.seed(1)

def test_getter():
    def fun(input_list):
        A = np.sum(input_list[0])
        B = np.sum(input_list[1])
        C = np.sum(input_list[1])
        return A + B + C

    d_fun = grad(fun)
    input_list = [npr.randn(5, 6),
                  npr.randn(4, 3),
                  npr.randn(2, 4)]

    result = d_fun(input_list)
    assert np.allclose(result[0], np.ones((5, 6)))
    assert np.allclose(result[1], 2 * np.ones((4, 3)))
    assert np.allclose(result[2], np.zeros((2, 4)))

def test_grads():
    def fun(input_list):
        A = np.sum(np.sin(input_list[0]))
        B = np.sum(np.cos(input_list[1]))
        return A + B

    def d_fun(input_list):
        g = grad(fun)(input_list)
        A = np.sum(g[0])
        B = np.sum(np.sin(g[0]))
        C = np.sum(np.sin(g[1]))
        return A + B + C

    input_list = [npr.randn(5, 6),
                  npr.randn(4, 3),
                  npr.randn(2, 4)]

    check_grads(fun, input_list)
    check_grads(d_fun, input_list)
