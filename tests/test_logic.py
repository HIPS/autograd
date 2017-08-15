from __future__ import division
import autograd.numpy as np
from autograd import grad
from autograd.util import *
from autograd.core import primitive_vjps, get_primitive

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
    fun = lambda x: np.real(x**2 if np.iscomplex(x) else np.sum(x))
    check_grads(fun, 2.)
    check_grads(fun, 2. + 1j)

def test_unimplemented_falseyness():
    def remove_grad_definitions(fun):
        return primitive_vjps[fun]

    def restore_grad_definitions(fun, grads):
        primitive_vjps[fun] = grads

    grad_defs = remove_grad_definitions(np.iscomplex)

    fun = lambda x: np.real(x**2 if np.iscomplex(x) else np.sum(x))
    check_grads(fun, 5.)
    check_grads(fun, 2. + 1j)

    restore_grad_definitions(np.iscomplex, grad_defs)
