"""
Test working properly with ufuncs in presence of other, non-differentiable array-like-containers.
Specifically, xarray.DataArray is tested.
"""

import numpy as onp
import pytest

import autograd.numpy as np
from autograd import grad
from autograd.test_util import check_grads

try:
    import xarray as xr
except ImportError:
    xr = None

requires_xarray = pytest.mark.skipif(xr is None, reason="xarray is required for this test")


@requires_xarray
def test_grad_through_dataarray_binary_ufunc():
    base = np.array([1.0, 2.0, 3.0])

    def f(x):
        out = x * xr.DataArray(base)
        return np.sum(out.data)

    assert f(2.0) == np.sum(2.0 * base)
    assert grad(f)(2.0) == np.sum(base)
    check_grads(f)(2.0)


@requires_xarray
def test_grad_through_dataarray_reversed_operand():
    base = np.array([2.2, 3.3, 4.4])

    def f(x):
        out = xr.DataArray(base) * x
        return np.sum(out.data)

    assert f(3.5) == np.sum(3.5 * base)
    assert grad(f)(3.5) == np.sum(base)
    check_grads(f)(3.5)


@requires_xarray
def test_grad_through_dataarray_unary_ufunc():
    base = np.array([-1.0, -2.0, -3.0])

    def f(x):
        out = np.abs(xr.DataArray(base) * x)
        return np.sum(out.data)

    assert f(2.0) == np.sum(np.abs(2.0 * base))
    # base is all negative so abs acts as negation here.
    assert grad(f)(2.0) == -np.sum(base)
    check_grads(f)(2.0)


@requires_xarray
def test_grad_through_dataarray_transcendental_ufunc():
    base = np.array([0.5, 1.0, 1.5])

    def f(x):
        out = np.sin(xr.DataArray(base) * x)
        return np.sum(out.data)

    x0 = 0.7
    assert onp.isclose(f(x0), np.sum(np.sin(base * x0)))
    assert onp.isclose(grad(f)(x0), np.sum(base * np.cos(base * x0)))
    check_grads(f)(x0)


@requires_xarray
def test_grad_through_two_dataarrays_holding_boxes():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])

    def f(x):
        da = xr.DataArray(a) * x
        db = xr.DataArray(b) * x
        out = da * db
        return np.sum(out.data)

    assert f(2.0) == 4.0 * np.sum(a * b)
    assert grad(f)(2.0) == 2.0 * 2.0 * np.sum(a * b)
    check_grads(f)(2.0)


@requires_xarray
def test_grad_through_dataarray_box_plus_plain_dataarray():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([10.0, 20.0, 30.0])

    def f(x):
        out = (xr.DataArray(a) * x) + xr.DataArray(b)
        return np.sum(out.data)

    assert f(2.0) == 2.0 * np.sum(a) + np.sum(b)
    assert grad(f)(2.0) == np.sum(a)
    check_grads(f)(2.0)


@requires_xarray
def test_grad_through_dataarray_box_times_plain_ndarray():
    a = np.array([1.0, 2.0, 3.0])
    c = np.array([2.0, 3.0, 4.0])

    def f(x):
        out = (xr.DataArray(a) * x) * c
        return np.sum(out.data)

    assert f(2.0) == 2.0 * np.sum(a * c)
    assert grad(f)(2.0) == np.sum(a * c)
    check_grads(f)(2.0)


@requires_xarray
def test_grad_through_dataarray_box_times_python_scalar():
    a = np.array([1.0, 2.0, 3.0])

    def f(x):
        # box-bearing DataArray * Python float.
        out = (xr.DataArray(a) * x) * 5.0
        return np.sum(out.data)

    assert f(2.0) == 2.0 * 5.0 * np.sum(a)
    assert grad(f)(2.0) == 5.0 * np.sum(a)
    check_grads(f)(2.0)


@requires_xarray
def test_grad_through_dataarray_division():
    base = np.array([0.5, 1.0, 1.5])

    def f(x):
        out = xr.DataArray(base) / x
        return np.sum(out.data)

    assert onp.isclose(f(2.0), np.sum(base) / 2.0)
    assert onp.isclose(grad(f)(2.0), -np.sum(base) / 4.0)
    check_grads(f)(2.0)


@requires_xarray
def test_grad_through_dataarray_exp():
    base = np.array([0.5, 1.0, 1.5])

    def f(x):
        out = np.exp(xr.DataArray(base) * x)
        return np.sum(out.data)

    x0 = 0.7
    assert onp.isclose(f(x0), np.sum(np.exp(base * x0)))
    assert onp.isclose(grad(f)(x0), np.sum(base * np.exp(base * x0)))
    check_grads(f)(x0)


@requires_xarray
def test_grad_through_dataarray_maximum():
    base = np.array([-1.0, 1.0, -2.0, 2.0])

    def f(x):
        # ReLU-style: maximum of a box-bearing DataArray and a Python scalar.
        out = np.maximum(xr.DataArray(base) * x, 0.0)
        return np.sum(out.data)

    x0 = 1.5  # no element of base*x0 is exactly 0
    assert onp.isclose(f(x0), np.sum(np.maximum(base * x0, 0.0)))
    # gradient picks up base only where base*x0 > 0
    assert onp.isclose(grad(f)(x0), np.sum(base * (base * x0 > 0)))
    check_grads(f)(x0)


@requires_xarray
def test_grad_through_dataarray_array_valued_box():
    coeffs = np.array([2.0, 3.0, 4.0])

    def f(x):
        out = xr.DataArray(coeffs) * x
        return np.sum(out.data)

    x0 = np.array([1.0, -1.0, 0.5])
    assert onp.isclose(f(x0), np.sum(coeffs * x0))
    assert onp.allclose(grad(f)(x0), coeffs)
    check_grads(f)(x0)


@requires_xarray
def test_grad_through_box_times_dataarray_holding_box():
    a = np.array([1.0, 2.0, 3.0])

    def f(x):
        da = xr.DataArray(a) * x
        out = x * da
        return np.sum(out.data)

    assert f(2.0) == 4.0 * np.sum(a)
    assert grad(f)(2.0) == 2.0 * 2.0 * np.sum(a)
    check_grads(f)(2.0)


# Regression tests


def test_ufunc_box_scalar_times_plain_ndarray():
    coeffs = np.array([1.0, 2.0, 3.0])

    def f(x):
        return np.sum(x * coeffs)

    assert f(2.0) == np.sum(2.0 * coeffs)
    assert grad(f)(2.0) == np.sum(coeffs)
    check_grads(f)(2.0)


def test_ufunc_plain_ndarray_times_box_scalar():
    coeffs = np.array([1.0, 2.0, 3.0])

    def f(x):
        return np.sum(coeffs * x)

    assert f(2.0) == np.sum(coeffs * 2.0)
    assert grad(f)(2.0) == np.sum(coeffs)
    check_grads(f)(2.0)


def test_ufunc_box_array_elementwise_plain_ndarray():
    coeffs = np.array([2.0, 3.0, 4.0])

    def f(x):
        return np.sum(x * coeffs)

    x0 = np.array([1.0, -1.0, 0.5])
    assert onp.allclose(f(x0), np.sum(x0 * coeffs))
    assert onp.allclose(grad(f)(x0), coeffs)
    check_grads(f)(x0)


def test_ufunc_between_two_arrayboxes_still_differentiates():
    x = np.array([1.0, 2.0, 3.0])
    g = grad(lambda v: np.sum(np.multiply(v, v)))(x)
    assert onp.allclose(g, 2.0 * x)


def test_ufunc_arraybox_and_scalar_still_differentiates():
    check_grads(lambda v: np.sum(np.sin(v) + 2.0 * v))(np.array([0.1, 0.2, 0.3]))
