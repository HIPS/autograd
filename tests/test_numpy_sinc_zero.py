import numpy as np
import pytest

import autograd.numpy as anp
from autograd import grad, make_jvp, make_vjp

NODES, WEIGHTS = np.polynomial.legendre.leggauss(32)
NODES = (NODES + 1.0) / 2.0
WEIGHTS = WEIGHTS / 2.0


def integral_derivative(value, order):
    """Differentiate sinc(x) = integral_0^1 cos(pi*x*t) dt analytically."""
    value = np.asarray(value)
    integrand = (np.pi * NODES) ** order * np.cos(np.pi * value[..., None] * NODES + order * np.pi / 2)
    return np.sum(WEIGHTS * integrand, axis=-1)


@pytest.mark.parametrize("shape", [(), (5,), (2, 3), (0,)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_sinc_zero_public_derivatives(shape, dtype):
    value = np.linspace(-0.7, 0.8, int(np.prod(shape))).reshape(shape).astype(dtype)
    if value.size:
        value.flat[value.size // 2] = 0.0
    if np.issubdtype(dtype, np.complexfloating):
        value = value + np.where(value == 0, 0, 0.15j).astype(dtype)
    cotangent = np.linspace(0.3, 1.7, value.size).reshape(shape).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        cotangent = cotangent + np.asarray(0.2j, dtype=dtype)
    expected = cotangent * integral_derivative(value, 1)
    with np.errstate(divide="raise", invalid="raise"):
        pullback, primal = make_vjp(anp.sinc)(value)
        np.testing.assert_allclose(primal, np.sinc(value), rtol=2e-6, atol=2e-6)
        vjp = pullback(cotangent)
        _, jvp = make_jvp(anp.sinc)(value)(cotangent)
        gradient = grad(lambda z: anp.real(anp.sum(anp.sinc(z) * cotangent)))(value)
    for actual in (vjp, jvp, gradient):
        assert actual.shape == value.shape and actual.dtype == value.dtype
        assert np.all(np.isfinite(actual))
        np.testing.assert_allclose(actual, expected, rtol=5e-6, atol=5e-6)


def nth_derivative(function, order, mode):
    for index in range(order):
        if mode == "rev" or (mode == "mixed" and index % 2 == 0):
            function = grad(function)
        else:
            previous = function
            function = lambda x, previous=previous: make_jvp(previous)(x)(1.0)[1]
    return function


@pytest.mark.parametrize("order", range(1, 7))
@pytest.mark.parametrize("mode", ["rev", "fwd", "mixed"])
def test_sinc_zero_higher_derivatives(order, mode):
    expected = integral_derivative(0.0, order)
    with np.errstate(divide="raise", invalid="raise"):
        actual = nth_derivative(anp.sinc, order, mode)(0.0)
    assert np.isfinite(actual)
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-12)


@pytest.mark.parametrize("order", [1, 3, 5])
def test_sinc_product_identity_at_zero(order):
    actual = nth_derivative(lambda x: x * anp.sinc(x), order, "mixed")(0.0)
    expected = np.pi ** (order - 1) * np.sin(order * np.pi / 2)
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-12)


@pytest.mark.parametrize("shape", [(3,), (2, 2)])
def test_sinc_mixed_zero_nonzero_coupled_hessian(shape):
    value = np.linspace(-0.7, 0.8, int(np.prod(shape))).reshape(shape)
    value.flat[1] = 0.0
    weights = np.linspace(0.3, 1.7, value.size).reshape(shape)
    tangent = np.linspace(-0.6, 0.5, value.size).reshape(shape)

    def objective(x):
        values = anp.sinc(x)
        return 0.5 * anp.sum(weights * values) ** 2 + 0.3 * anp.sum(values**2)

    values = np.sinc(value)
    first, second = integral_derivative(value, 1), integral_derivative(value, 2)
    expected = weights * first * np.sum(weights * first * tangent)
    expected += 0.6 * first**2 * tangent
    expected += (np.sum(weights * values) * weights + 0.6 * values) * second * tangent
    forward = make_jvp(grad(objective))(value)(tangent)[1]
    reverse = grad(lambda x: anp.sum(grad(objective)(x) * tangent))(value)
    for actual in (forward, reverse):
        assert np.all(np.isfinite(actual)) and actual.shape == value.shape
        np.testing.assert_allclose(actual, expected, rtol=3e-11, atol=3e-11)


def test_sinc_noncontiguous_mixed_input():
    value = np.array([-0.8, 0.0, 0.4, 1.2])[::-1]
    weights = np.array([0.3, 0.7, 1.1, 1.4])
    actual = grad(lambda x: anp.sum(anp.sinc(x) * weights))(value)
    np.testing.assert_allclose(actual, integral_derivative(value, 1) * weights, rtol=2e-12, atol=2e-12)


def test_sinc_zero_cotangent_and_tangent_remain_finite():
    value = np.array([0.0, 0.4])
    zeros = np.zeros_like(value)
    pullback, _ = make_vjp(anp.sinc)(value)
    for actual in (pullback(zeros), make_jvp(anp.sinc)(value)(zeros)[1]):
        assert np.all(np.isfinite(actual))
        np.testing.assert_array_equal(actual, zeros)
