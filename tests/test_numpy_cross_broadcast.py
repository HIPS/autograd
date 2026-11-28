import numpy as np
import pytest

import autograd.numpy as anp
from autograd import grad, make_jvp, make_vjp
from autograd.test_util import check_grads


def cross_operand(module, value, other, argnum, kwargs):
    if argnum == 0:
        return module.cross(value, other, **kwargs)
    return module.cross(other, value, **kwargs)


def numpy_basis_vjp(value, other, argnum, kwargs, weights):
    expected = np.empty_like(value)
    for index in np.ndindex(value.shape):
        basis = np.zeros_like(value)
        basis[index] = 1
        expected[index] = np.sum(cross_operand(np, basis, other, argnum, kwargs) * weights)
    return expected


@pytest.mark.parametrize("argnum", [0, 1])
@pytest.mark.parametrize(
    ("a_shape", "b_shape", "kwargs"),
    [
        ((3,), (3,), {}),
        ((2, 3), (2, 3), {}),
        ((3,), (2, 3), {}),
        ((1, 3), (2, 3), {}),
        ((2, 1, 3), (1, 4, 3), {}),
        ((3,), (2, 3), {"axisa": 0, "axisb": 1, "axisc": 1}),
        ((3,), (3, 2), {"axisa": 0, "axisb": 0, "axisc": 0}),
        ((3, 1), (3, 4), {"axisa": 0, "axisb": 0, "axisc": 0}),
        ((2, 3, 1), (1, 3, 4), {"axis": 1}),
        ((2, 3), (2, 3), {"axis": 1}),
        ((2, 3, 4), (2, 4, 3), {"axisa": 1, "axisb": -1, "axisc": 0}),
        ((3, 2, 4), (1, 3, 4), {"axisa": 0, "axisb": 1, "axisc": -1}),
    ],
)
def test_cross_broadcast_vjp(a_shape, b_shape, kwargs, argnum):
    a = np.linspace(0.2, 1.7, np.prod(a_shape)).reshape(a_shape)
    b = np.linspace(-1.3, 0.9, np.prod(b_shape)).reshape(b_shape)
    value, other = (a, b) if argnum == 0 else (b, a)
    function = lambda z: cross_operand(anp, z, other, argnum, kwargs)
    expected_value = cross_operand(np, value, other, argnum, kwargs)
    weights = np.linspace(0.3, 1.9, expected_value.size).reshape(expected_value.shape)
    pullback, actual_value = make_vjp(function)(value)
    np.testing.assert_array_equal(actual_value, expected_value)
    expected = numpy_basis_vjp(value, other, argnum, kwargs, weights)
    for actual in (pullback(weights), grad(lambda z: anp.sum(function(z) * weights))(value)):
        assert actual.shape == value.shape
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    tangent = np.linspace(-0.2, 0.6, value.size).reshape(value.shape)
    _, actual_tangent = make_jvp(function)(value)(tangent)
    np.testing.assert_allclose(
        actual_tangent, cross_operand(np, tangent, other, argnum, kwargs), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("argnum", [0, 1])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_cross_broadcast_dtype(dtype, argnum):
    value = np.linspace(0.2, 1.7, 3).astype(dtype)
    other = np.linspace(-1.3, 0.9, 6).reshape(2, 3).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        value = value + np.asarray(0.3j, dtype=dtype)
        other = other + np.asarray(-0.4j, dtype=dtype)
    function = lambda z: cross_operand(anp, z, other, argnum, {})
    weights = np.arange(1, 7, dtype=dtype).reshape(2, 3)
    actual = grad(lambda z: anp.real(anp.sum(function(z) * weights)))(value)
    assert actual.shape == value.shape
    assert actual.dtype == value.dtype
    np.testing.assert_allclose(
        actual, numpy_basis_vjp(value, other, argnum, {}, weights), rtol=2e-6, atol=2e-6
    )


@pytest.mark.parametrize("argnum", [0, 1])
@pytest.mark.parametrize("layout", ["fortran", "reversed", "transposed"])
def test_cross_broadcast_layout(layout, argnum):
    a = np.linspace(0.2, 1.7, 6).reshape(2, 1, 3)
    b = np.linspace(-1.3, 0.9, 12).reshape(1, 4, 3)
    kwargs = {}
    if layout == "fortran":
        a, b = np.asfortranarray(a), np.asfortranarray(b)
    elif layout == "reversed":
        a, b = a[..., ::-1], b[..., ::-1]
    else:
        a, b = a.transpose(2, 0, 1), b.transpose(2, 0, 1)
        kwargs = {"axis": 0}
    value, other = (a, b) if argnum == 0 else (b, a)
    result = cross_operand(np, value, other, argnum, kwargs)
    weights = np.linspace(0.3, 1.9, result.size).reshape(result.shape)
    actual = grad(lambda z: anp.sum(cross_operand(anp, z, other, argnum, kwargs) * weights))(value)
    assert actual.shape == value.shape
    np.testing.assert_allclose(
        actual, numpy_basis_vjp(value, other, argnum, kwargs, weights), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("argnum", [0, 1])
@pytest.mark.parametrize(
    ("a_shape", "b_shape", "kwargs"),
    [
        ((3,), (2, 3), {}),
        ((2, 1, 3), (1, 2, 3), {}),
        ((2, 3, 1), (1, 3, 2), {"axis": 1}),
    ],
)
def test_cross_broadcast_higher_order(a_shape, b_shape, kwargs, argnum):
    a = np.linspace(0.2, 0.8, np.prod(a_shape)).reshape(a_shape)
    b = np.linspace(-0.6, 0.9, np.prod(b_shape)).reshape(b_shape)
    value, other = (a, b) if argnum == 0 else (b, a)
    check_grads(
        lambda z: anp.sum(anp.sin(cross_operand(anp, z, other, argnum, kwargs)) ** 2),
        modes=["fwd", "rev"],
        order=2,
    )(value)


@pytest.mark.parametrize("argnum", [0, 1])
def test_cross_broadcast_positional_axes(argnum):
    a = np.linspace(0.2, 1.7, 3)
    b = np.linspace(-1.3, 0.9, 6).reshape(2, 3)
    value = a if argnum == 0 else b

    def function(module, z):
        return module.cross(z, b, 0, 1, 0) if argnum == 0 else module.cross(a, z, 0, 1, 0)

    result = function(np, value)
    weights = np.linspace(0.3, 1.9, result.size).reshape(result.shape)
    expected = np.empty_like(value)
    for index in np.ndindex(value.shape):
        basis = np.zeros_like(value)
        basis[index] = 1
        expected[index] = np.sum(function(np, basis) * weights)
    actual = grad(lambda z: anp.sum(function(anp, z) * weights))(value)
    assert actual.shape == value.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
