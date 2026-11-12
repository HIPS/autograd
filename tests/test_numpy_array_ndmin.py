import numpy as onp
import pytest
from numpy.testing import assert_allclose

import autograd.numpy as np
from autograd import grad, make_jvp, make_vjp


def assert_constructor_derivatives(build, x, ndmin):
    fun = lambda value: np.array(build(value), ndmin=ndmin)
    numpy_fun = lambda value: onp.array(build(value), ndmin=ndmin)
    expected_value = numpy_fun(x)
    weights = onp.arange(expected_value.size, dtype=float).reshape(expected_value.shape) + 1.0
    tangent = onp.arange(x.size, dtype=float).reshape(x.shape) * 0.1 + 0.37
    epsilon = 1e-5
    expected_grad = onp.empty_like(x)
    for index in onp.ndindex(x.shape):
        step = onp.zeros_like(x)
        step[index] = epsilon
        expected_grad[index] = (
            onp.sum(numpy_fun(x + step) * weights) - onp.sum(numpy_fun(x - step) * weights)
        ) / (2 * epsilon)
    expected_jvp = (numpy_fun(x + epsilon * tangent) - numpy_fun(x - epsilon * tangent)) / (2 * epsilon)

    vjp, value = make_vjp(fun)(x)
    actual_grad = vjp(weights)
    actual_jvp = make_jvp(fun)(x)(tangent)[1]
    assert value.shape == expected_value.shape
    assert_allclose(value, expected_value)
    assert onp.shape(actual_grad) == x.shape
    assert_allclose(actual_grad, expected_grad, rtol=1e-6, atol=1e-7)
    assert onp.shape(actual_jvp) == expected_value.shape
    assert_allclose(actual_jvp, expected_jvp, rtol=1e-6, atol=1e-7)
    assert_allclose(grad(lambda value: np.sum(fun(value) * weights))(x), expected_grad, rtol=1e-6)
    assert_allclose(onp.sum(actual_grad * tangent), onp.sum(actual_jvp * weights), atol=1e-7)


LEAVES = [
    pytest.param(onp.array(1.0), 2, id="scalar"),
    pytest.param(onp.array([1.0, 2.0]), 4, id="vector-many-leading-axes"),
    pytest.param(onp.array([[1.0, 2.0]]), 4, id="matrix-singleton-axis"),
    pytest.param(onp.arange(12.0).reshape(3, 4).T[:, ::2], 5, id="noncontiguous"),
    pytest.param(onp.empty((0, 2)), 4, id="empty-leaf"),
]


@pytest.mark.parametrize("container", [list, tuple])
@pytest.mark.parametrize("active", [0, 1])
@pytest.mark.parametrize("x,ndmin", LEAVES)
def test_array_ndmin_sequence_leaf(container, active, x, ndmin):
    constant = onp.full(x.shape, 3.0)

    def build(value):
        values = [constant, constant]
        values[active] = value
        return container(values)

    assert_constructor_derivatives(build, x, ndmin)


@pytest.mark.parametrize("container", [list, tuple])
@pytest.mark.parametrize("ndmin", [0, 2])
def test_array_ndmin_sequence_without_promotion(container, ndmin):
    x = onp.array([1.0, 2.0])
    assert_constructor_derivatives(lambda value: container([onp.array([3.0, 4.0]), value]), x, ndmin)


@pytest.mark.parametrize(
    "x,ndmin", [pytest.param(onp.array(1.0), 3), pytest.param(onp.array([[1.0, 2.0]]), 4)]
)
def test_array_ndmin_direct_input_control(x, ndmin):
    assert_constructor_derivatives(lambda value: value, x, ndmin)


@pytest.mark.parametrize("nested", [False, True])
def test_array_ndmin_repeated_leaf(nested):
    x = onp.array([1.0, 2.0])

    def build(value):
        return [[value, value], [value, value]] if nested else (value, value)

    assert_constructor_derivatives(build, x, 5)


@pytest.mark.parametrize("container", [list, tuple])
@pytest.mark.parametrize("shape", [(), (1, 2)])
def test_array_ndmin_higher_order(container, shape):
    x = onp.arange(onp.prod(shape) if shape else 1, dtype=float).reshape(shape) + 1.0
    tangent = onp.full(shape, 0.37)
    constant = onp.full(shape, 3.0)

    def objective(value):
        result = np.array(container([constant, value, value]), ndmin=len(shape) + 3)
        return np.sum(np.sin(result) + result**3)

    expected_grad = 2 * (onp.cos(x) + 3 * x**2)
    expected_hvp = 2 * (-onp.sin(x) + 6 * x) * tangent
    actual_grad = grad(objective)(x)
    assert onp.shape(actual_grad) == x.shape
    assert_allclose(actual_grad, expected_grad)
    rr = grad(lambda value: np.sum(grad(objective)(value) * tangent))(x)
    fr = make_jvp(grad(objective))(x)(tangent)[1]
    assert onp.shape(rr) == x.shape
    assert onp.shape(fr) == x.shape
    assert_allclose(rr, expected_hvp)
    assert_allclose(fr, expected_hvp)
