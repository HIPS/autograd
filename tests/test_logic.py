import warnings
from contextlib import contextmanager

import pytest

import autograd.numpy as np
from autograd import deriv, grad
from autograd.core import primitive_vjps
from autograd.extend import primitive
from autograd.test_util import check_grads


def test_assert():
    # from https://github.com/HIPS/autograd/issues/43
    def fun(x):
        assert np.allclose(x, (x * 3.0) / 3.0)
        return np.sum(x)

    check_grads(fun)(np.array([1.0, 2.0, 3.0]))


def test_nograd():
    # we want this to raise non-differentiability error
    fun = lambda x: np.allclose(x, (x * 3.0) / 3.0)
    with pytest.raises(TypeError):
        with warnings.catch_warnings(record=True) as w:
            grad(fun)(np.array([1.0, 2.0, 3.0]))


def test_no_vjp_def():
    fun = primitive(lambda x: 2.0 * x)
    with pytest.raises(NotImplementedError):
        grad(fun)(1.0)


def test_no_jvp_def():
    fun = primitive(lambda x: 2.0 * x)
    with pytest.raises(NotImplementedError):
        deriv(fun)(1.0)


def test_falseyness():
    fun = lambda x: np.real(x**2 if np.iscomplex(x) else np.sum(x))
    check_grads(fun)(2.0)
    check_grads(fun)(2.0 + 1j)


def test_unimplemented_falseyness():
    @contextmanager
    def remove_grad_definitions(fun):
        vjpmaker = primitive_vjps.pop(fun, None)
        yield
        if vjpmaker:
            primitive_vjps[fun] = vjpmaker

    with remove_grad_definitions(np.iscomplex):
        fun = lambda x: np.real(x**2 if np.iscomplex(x) else np.sum(x))
        check_grads(fun)(5.0)
        check_grads(fun)(2.0 + 1j)
