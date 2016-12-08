from __future__ import division
import autograd.numpy as np
from autograd import grad
from autograd.util import *

def test_assert():
    # from https://github.com/HIPS/autograd/issues/43
    def fun(x):
        assert np.allclose(x, (x*3.0)/3.0)
        return np.sum(x)
    check_grads(fun, np.array([1.0, 2.0, 3.0]))

def test_nograd():
    # we want this to raise non-differentiability error
    fun = lambda x: np.allclose(x, (x*3.0)/3.0)
    try:
        grad(fun)(np.array([1., 2., 3.]))
    except TypeError:
        pass
    else:
        raise Exception('Expected non-differentiability exception')

def test_falseyness():
    fun = lambda x: x**2 if np.iscomplex(x) else np.sum(x)
    check_grads(fun, 2.)
    check_grads(fun, 2. + 1j)

def test_unimplemented_falseyness():
    def remove_grad_definitions(fun):
        grads, zero_vjps = fun.vjps, fun.zero_vjps
        fun.vjps, fun.zero_vjps = {}, set()
        return grads, zero_vjps

    def restore_grad_definitions(fun, grad_defs):
        fun.vjps, fun.zero_vjps = grad_defs

    grad_defs = remove_grad_definitions(np.iscomplex)

    fun = lambda x: x**2 if np.iscomplex(x) else np.sum(x)
    check_grads(fun, 5.)
    check_grads(fun, 2. + 1j)

    restore_grad_definitions(np.iscomplex, grad_defs)
