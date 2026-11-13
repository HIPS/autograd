import numpy as onp
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import autograd.numpy as np
from autograd import grad, make_jvp, make_vjp


def complex_input(dtype, layout):
    if layout == "scalar-real":
        return onp.array(complex(2.0, onp.nan), dtype=dtype)
    if layout == "scalar-imag":
        return onp.array(complex(onp.nan, 2.0), dtype=dtype)
    if layout == "empty":
        return onp.empty((0, 2), dtype=dtype)
    values = onp.array(
        [
            complex(1.0, onp.nan),
            complex(2.0, onp.inf),
            complex(-3.0, -onp.inf),
            complex(onp.nan, 4.0),
            complex(onp.inf, -5.0),
            complex(-onp.inf, 6.0),
            complex(onp.nan, onp.inf),
            complex(7.0, 8.0),
        ],
        dtype=dtype,
    )
    return values.reshape(2, 4).T[::2] if layout == "noncontiguous" else values


@pytest.mark.parametrize("dtype", [onp.complex64, onp.complex128])
@pytest.mark.parametrize("layout", ["scalar-real", "scalar-imag", "vector", "noncontiguous", "empty"])
def test_nan_to_num_complex_components(dtype, layout):
    x = complex_input(dtype, layout)
    tangent = onp.empty_like(x)
    tangent.real = 0.2
    tangent.imag = -0.3
    weights = onp.empty_like(x)
    weights.real = onp.arange(x.size).reshape(x.shape) * 0.1 + 0.7
    weights.imag = onp.arange(x.size).reshape(x.shape) * 0.1 - 0.4
    expected_vjp = onp.empty_like(weights)
    expected_vjp.real = onp.where(onp.isfinite(x.real), weights.real, 0.0)
    expected_vjp.imag = onp.where(onp.isfinite(x.imag), weights.imag, 0.0)
    expected_jvp = onp.empty_like(tangent)
    expected_jvp.real = onp.where(onp.isfinite(x.real), tangent.real, 0.0)
    expected_jvp.imag = onp.where(onp.isfinite(x.imag), tangent.imag, 0.0)

    vjp, value = make_vjp(np.nan_to_num)(x)
    actual_vjp = vjp(weights)
    actual_jvp = make_jvp(np.nan_to_num)(x)(tangent)[1]
    assert_array_equal(value, onp.nan_to_num(x))
    assert value.shape == actual_vjp.shape == actual_jvp.shape == x.shape
    assert value.dtype == actual_vjp.dtype == actual_jvp.dtype == x.dtype
    assert_array_equal(actual_vjp, expected_vjp)
    assert_array_equal(actual_jvp, expected_jvp)

    # Perturb each coordinate independently, retaining its nonfinite counterpart.
    epsilon = 1e-2 if dtype == onp.complex64 else 1e-5
    plus, minus = x.copy(), x.copy()
    plus.real = x.real + epsilon * tangent.real
    minus.real = x.real - epsilon * tangent.real
    plus.imag = x.imag + epsilon * tangent.imag
    minus.imag = x.imag - epsilon * tangent.imag
    numerical = (onp.nan_to_num(plus) - onp.nan_to_num(minus)) / (2 * epsilon)
    assert_allclose(
        actual_jvp,
        numerical,
        rtol=1e-3 if dtype == onp.complex64 else 1e-7,
        atol=1e-4 if dtype == onp.complex64 else 1e-8,
    )
    assert_allclose(onp.real(onp.sum(actual_vjp * tangent)), onp.real(onp.sum(weights * actual_jvp)))


@pytest.mark.parametrize("dtype", [onp.complex64, onp.complex128])
@pytest.mark.parametrize("scalar", [False, True])
def test_nan_to_num_finite_complex_control(dtype, scalar):
    x = onp.array(complex(1.0, 2.0) if scalar else [complex(1.0, 2.0), complex(-2.0, 0.7)], dtype=dtype)
    direction = onp.full_like(x, complex(0.2, -0.3))
    weights = onp.full_like(x, complex(0.7, -0.4))
    vjp, value = make_vjp(np.nan_to_num)(x)
    assert_array_equal(value, x)
    assert_array_equal(vjp(weights), weights)
    assert_array_equal(make_jvp(np.nan_to_num)(x)(direction)[1], direction)


@pytest.mark.parametrize("dtype", [onp.float32, onp.float64])
def test_nan_to_num_real_control(dtype):
    x = onp.array([onp.nan, onp.inf, -onp.inf, 1.0, -2.0], dtype=dtype)
    direction = onp.full_like(x, 0.2)
    weights = onp.full_like(x, 0.7)
    vjp, value = make_vjp(np.nan_to_num)(x)
    assert_array_equal(value, onp.nan_to_num(x))
    assert_array_equal(vjp(weights), onp.array([0.0, 0.0, 0.0, 0.7, 0.7], dtype=dtype))
    assert_array_equal(make_jvp(np.nan_to_num)(x)(direction)[1], onp.array([0, 0, 0, 0.2, 0.2], dtype=dtype))


@pytest.mark.parametrize("component", ["real", "imag"])
@pytest.mark.parametrize("bad", [onp.nan, onp.inf, -onp.inf])
def test_nan_to_num_surviving_real_parameter_higher_order(component, bad):
    if component == "real":
        constant = complex(2.0, bad)
        linear = lambda t: np.real(np.nan_to_num(constant + t))
        numpy_linear = lambda t: onp.real(onp.nan_to_num(constant + t))
    else:
        constant = complex(bad, 2.0)
        linear = lambda t: np.imag(np.nan_to_num(constant + 1j * t))
        numpy_linear = lambda t: onp.imag(onp.nan_to_num(constant + 1j * t))
    squared = lambda t: linear(t) ** 2
    numpy_squared = lambda t: numpy_linear(t) ** 2
    point = 0.0
    epsilon = 1e-4
    numerical_gradient = (numpy_squared(point + epsilon) - numpy_squared(point - epsilon)) / (2 * epsilon)
    numerical_hessian = (
        numpy_squared(point + epsilon) - 2 * numpy_squared(point) + numpy_squared(point - epsilon)
    ) / epsilon**2
    assert linear(point) == numpy_linear(point) == 2.0
    assert_allclose(grad(linear)(point), 1.0)
    assert_allclose(make_jvp(linear)(point)(0.3)[1], 0.3)
    assert_allclose(grad(squared)(point), 4.0)
    assert_allclose(grad(squared)(point), numerical_gradient)
    assert_allclose(grad(grad(squared))(point), 2.0)
    assert_allclose(grad(grad(squared))(point), numerical_hessian, rtol=1e-6)
    assert_allclose(make_jvp(grad(squared))(point)(0.3)[1], 0.6)
