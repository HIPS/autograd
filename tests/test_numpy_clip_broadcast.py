import numpy as onp
import pytest
from numpy.testing import assert_allclose

import autograd.numpy as np
from autograd import grad, make_jvp, make_vjp

CLIP_CASES = [
    pytest.param(onp.array([-0.6, 0.2, 1.4]), -0.2, 1.0, id="scalar-bounds-control"),
    pytest.param(
        onp.array([-0.6, 0.2, 1.4]),
        onp.array([-0.8, -0.3, 0.8]),
        onp.array([-0.4, 0.6, 1.2]),
        id="same-shape-control",
    ),
    pytest.param(0.4, None, onp.array([0.1, 0.7, 2.0]), id="scalar-to-vector"),
    pytest.param(onp.array([-0.4, 0.6]), onp.array([[-1.0], [0.2], [0.8]]), None, id="upper-none"),
    pytest.param(
        onp.array([0.4, 1.4]),
        onp.array([[-0.2], [0.7], [-1.0]]),
        onp.array([[1.0, 2.0], [2.0, 1.0], [0.2, 3.0]]),
        id="vector-to-matrix",
    ),
    pytest.param(
        onp.array([[-0.4], [0.6]]),
        onp.array([[-1.0, 0.0, 0.8]]),
        onp.array([[0.1, 1.0, 2.0]]),
        id="singleton-axis",
    ),
    pytest.param(
        onp.array([[[-0.4], [0.6]]]),
        onp.array([[[-1.0, 0.0, 0.8]], [[-0.7, 0.2, 1.0]]]),
        onp.array([[[0.1, 1.0, 2.0]], [[0.3, 1.1, 2.2]]]),
        id="leading-and-singleton-axes",
    ),
    pytest.param(
        onp.array([[-0.4, 0.1, 0.6], [0.6, 0.9, 1.3]])[:, ::2],
        onp.array([[[-1.0, 0.0]], [[0.2, 0.8]], [[-0.7, 0.3]]]),
        1.0,
        id="noncontiguous-input",
    ),
    pytest.param(onp.array([[0.2, 0.6, 1.2]]), onp.empty((0, 3)), 2.0, id="empty-broadcast"),
    pytest.param(onp.array(0.4), 0.0, onp.empty((0,)), id="scalar-empty-broadcast"),
]


def _assert_array(actual, expected):
    assert onp.shape(actual) == onp.shape(expected)
    assert_allclose(actual, expected, rtol=1e-7, atol=1e-8)


def _basis_pullback(x, values):
    result = onp.empty_like(onp.asarray(x))
    for index in onp.ndindex(result.shape):
        basis = onp.zeros_like(result)
        basis[index] = 1.0
        result[index] = onp.sum(values * onp.broadcast_to(basis, values.shape))
    return result


def _interior_mask(x, lower, upper):
    value = onp.clip(x, lower, upper)
    broadcast_x = onp.broadcast_to(x, value.shape)
    mask = onp.ones(value.shape, dtype=bool)
    if lower is not None:
        assert onp.all(broadcast_x != lower)
        mask &= broadcast_x > lower
    if upper is not None:
        assert onp.all(broadcast_x != upper)
        mask &= broadcast_x < upper
    return value, mask


def _finite_gradient(fun, x):
    result = onp.empty_like(onp.asarray(x))
    for index in onp.ndindex(result.shape):
        delta = onp.zeros_like(result)
        delta[index] = 1e-5
        result[index] = (fun(onp.asarray(x) + delta) - fun(onp.asarray(x) - delta)) / 2e-5
    return result


@pytest.mark.parametrize("x, lower, upper", CLIP_CASES)
def test_clip_broadcast_input_gradient(x, lower, upper):
    expected_value, mask = _interior_mask(x, lower, upper)
    weights = (onp.arange(expected_value.size) + 1.0).reshape(expected_value.shape) / 3.0
    expected_gradient = _basis_pullback(x, weights * mask)
    finite_gradient = _finite_gradient(lambda a: onp.sum(weights * onp.clip(a, lower, upper)), x)
    _assert_array(expected_gradient, finite_gradient)

    def fun(a):
        return np.clip(a, lower, upper)

    pullback, value = make_vjp(fun)(x)
    _assert_array(value, expected_value)
    _assert_array(pullback(weights), expected_gradient)
    _assert_array(grad(lambda a: np.sum(weights * fun(a)))(x), expected_gradient)

    tangent = onp.linspace(0.2, 0.8, onp.size(x)).reshape(onp.shape(x))
    if isinstance(x, float):
        tangent = tangent.item()
    expected_jvp = onp.broadcast_to(tangent, expected_value.shape) * mask
    finite_jvp = (
        onp.clip(onp.asarray(x) + 1e-5 * tangent, lower, upper)
        - onp.clip(onp.asarray(x) - 1e-5 * tangent, lower, upper)
    ) / 2e-5
    _assert_array(expected_jvp, finite_jvp)
    value, derivative = make_jvp(fun)(x)(tangent)
    _assert_array(value, expected_value)
    _assert_array(derivative, expected_jvp)
    assert_allclose(onp.sum(expected_gradient * tangent), onp.sum(weights * expected_jvp))


@pytest.mark.parametrize("x, lower, upper", [CLIP_CASES[i] for i in (2, 4, 5, 6)])
def test_clip_broadcast_nonlinear_second_derivative(x, lower, upper):
    value, mask = _interior_mask(x, lower, upper)
    weights = (onp.arange(value.size) + 1.0).reshape(value.shape) / 3.0
    tangent = onp.linspace(0.2, 0.8, onp.size(x)).reshape(onp.shape(x))
    if isinstance(x, float):
        tangent = tangent.item()

    def objective(a):
        return np.sum(weights * np.clip(a, lower, upper) ** 3 / 3.0)

    expected_gradient = _basis_pullback(x, weights * value**2 * mask)
    expected_hvp = _basis_pullback(x, 2.0 * weights * value * mask * onp.broadcast_to(tangent, value.shape))
    finite_gradient = _finite_gradient(lambda a: onp.sum(weights * onp.clip(a, lower, upper) ** 3 / 3.0), x)
    _assert_array(expected_gradient, finite_gradient)

    def numpy_gradient(a):
        current, interior = _interior_mask(a, lower, upper)
        return _basis_pullback(a, weights * current**2 * interior)

    finite_hvp = (
        numpy_gradient(onp.asarray(x) + 1e-5 * tangent) - numpy_gradient(onp.asarray(x) - 1e-5 * tangent)
    ) / 2e-5
    _assert_array(expected_hvp, finite_hvp)
    _assert_array(grad(objective)(x), expected_gradient)
    reverse_hvp = grad(lambda a: np.sum(grad(objective)(a) * tangent))(x)
    gradient, forward_hvp = make_jvp(grad(objective))(x)(tangent)
    _assert_array(gradient, expected_gradient)
    _assert_array(reverse_hvp, expected_hvp)
    _assert_array(forward_hvp, expected_hvp)


@pytest.mark.parametrize(
    "x, lower, upper, mask",
    [
        (0.5, onp.array([0.5, -0.1, 0.2]), onp.array([1.0, 0.5, 0.8]), onp.array([0.0, 0.0, 1.0])),
        (onp.array([0.0, 0.5, 1.0]), 0.0, 1.0, onp.array([0.0, 1.0, 0.0])),
    ],
)
def test_clip_broadcast_preserves_zero_gradient_at_ties(x, lower, upper, mask):
    weights = (onp.arange(mask.size) + 1.0).reshape(mask.shape) / 3.0
    expected = _basis_pullback(x, weights * mask)
    pullback, _ = make_vjp(lambda a: np.clip(a, lower, upper))(x)
    _assert_array(pullback(weights), expected)
    _assert_array(grad(lambda a: np.sum(weights * np.clip(a, lower, upper)))(x), expected)
    tangent = 1.0 if isinstance(x, float) else onp.ones_like(x)
    _, derivative = make_jvp(lambda a: np.clip(a, lower, upper))(x)(tangent)
    _assert_array(derivative, mask)
