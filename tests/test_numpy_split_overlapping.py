import numpy as onp
import pytest
from numpy.testing import assert_allclose

import autograd.numpy as np
from autograd import grad, make_jvp, make_vjp

FAMILIES = [
    ("split", (2, 5), -1),
    ("array_split", (2, 5), -1),
    ("vsplit", (5, 2), 0),
    ("hsplit", (2, 5), 1),
    ("dsplit", (2, 2, 5), 2),
]
LINEAR_CASES = [
    pytest.param(name, shape, [-1, 1], axis, "overlap", id=f"{name}-sorted-mixed-sign")
    for name, shape, axis in FAMILIES
] + [
    pytest.param("split", (2, 5), [2, 1], 1, "overlap", id="positive-nonmonotone"),
    pytest.param("array_split", (2, 5), [10, -10], 1, "overlap", id="clipped-overlap"),
    pytest.param("split", (2, 6), onp.array([2, 4]), 1, "partition", id="numpy-cuts"),
    pytest.param("array_split", (2, 6), (-4, -2), -1, "partition", id="negative-partition"),
    pytest.param("hsplit", (2, 6), [2, 2], 1, "partition", id="repeated-cut"),
    pytest.param("vsplit", (6, 2), [], 0, "partition", id="no-cuts"),
    pytest.param("dsplit", (2, 2, 6), (-10, 10), 2, "partition", id="clipped-partition"),
    pytest.param("split", (2, 6), 3, -1, "sections", id="equal-sections"),
    pytest.param("vsplit", (6, 2), onp.int64(3), 0, "sections", id="numpy-int-sections"),
    pytest.param("hsplit", (2, 6), onp.array(3), 1, "sections", id="zero-dimensional-sections"),
    pytest.param("array_split", (2, 5), 3, 1, "sections", id="uneven-sections"),
    pytest.param("array_split", (2, 2), 5, -1, "sections", id="more-sections-than-elements"),
    pytest.param("dsplit", (2, 2, 6), 3, 2, "sections", id="depth-sections"),
    pytest.param("split", (2, 0), [-1, 1], -1, "empty", id="empty-split-axis"),
    pytest.param("array_split", (0, 5), [-1, 1], 1, "empty", id="other-empty-axis"),
    pytest.param("split", (2, 5), [-1, 1], 1, "noncontiguous", id="noncontiguous-input"),
]


def _split(module, name, x, cuts, axis):
    if name in ("split", "array_split"):
        return getattr(module, name)(x, cuts, axis=axis)
    return getattr(module, name)(x, cuts)


def _assert_array(actual, expected, tolerance=1e-7):
    assert onp.shape(actual) == onp.shape(expected)
    assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


def _assert_parts(actual, expected):
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        _assert_array(a, b)


def _weights(parts, dtype):
    weights, offset = [], 1
    for part in parts:
        positions = onp.arange(offset, offset + part.size).reshape(part.shape)
        value = (-1.0) ** positions * (0.25 + positions / 10.0)
        if onp.issubdtype(dtype, onp.complexfloating):
            value = value + 1j * (0.1 + positions / 20.0)
        weights.append(value.astype(dtype))
        offset += part.size
    return weights


def _scatter(x, labels, values):
    result = onp.zeros(onp.shape(x), dtype=values[0].dtype)
    for indices, value in zip(labels, values):
        onp.add.at(result.reshape(-1), indices.reshape(-1), value.reshape(-1))
    return result


def _finite_gradient(fun, x):
    result = onp.empty_like(x)
    for index in onp.ndindex(x.shape):
        delta = onp.zeros_like(x)
        delta[index] = 1e-5
        result[index] = (fun(x + delta) - fun(x - delta)) / 2e-5
    return result


def _check_linear(name, x, cuts, axis, finite_difference=True):
    parts = _split(onp, name, x, cuts, axis)
    labels = _split(onp, name, onp.arange(x.size).reshape(x.shape), cuts, axis)
    weights = _weights(parts, x.dtype)
    expected = _scatter(x, labels, weights)

    def fun(a):
        return _split(np, name, a, cuts, axis)

    pullback, primal = make_vjp(fun)(x)
    _assert_parts(primal, parts)
    actual = pullback(weights)
    _assert_array(actual, expected)
    assert actual.dtype == expected.dtype
    objective = lambda a: np.real(sum(np.sum(w * p) for w, p in zip(weights, fun(a))))
    _assert_array(grad(objective)(x), expected)
    tangent = onp.linspace(-0.4, 0.7, x.size).reshape(x.shape).astype(x.dtype)
    if onp.iscomplexobj(x):
        tangent += 0.2j
    value, derivative = make_jvp(fun)(x)(tangent)
    _assert_parts(value, parts)
    expected_jvp = _split(onp, name, tangent, cuts, axis)
    _assert_parts(derivative, expected_jvp)
    assert_allclose(
        onp.sum(expected * tangent),
        sum(onp.sum(w * p) for w, p in zip(weights, derivative)),
        rtol=1e-6 if x.dtype == onp.float32 else 1e-7,
        atol=1e-7,
    )
    if finite_difference and x.size:
        numpy_objective = lambda a: sum(
            onp.sum(w * p) for w, p in zip(weights, _split(onp, name, a, cuts, axis))
        )
        _assert_array(expected, _finite_gradient(numpy_objective, x))
        plus = _split(onp, name, x + 1e-5 * tangent, cuts, axis)
        minus = _split(onp, name, x - 1e-5 * tangent, cuts, axis)
        _assert_parts(derivative, [(a - b) / 2e-5 for a, b in zip(plus, minus)])


@pytest.mark.parametrize("name, shape, cuts, axis, kind", LINEAR_CASES)
def test_split_overlap_and_partition(name, shape, cuts, axis, kind):
    if kind == "noncontiguous":
        x = onp.linspace(0.3, 1.2, 20).reshape(2, 10)[:, ::2]
        assert not x.flags.c_contiguous
    else:
        x = onp.linspace(0.3, 1.2, onp.prod(shape)).reshape(shape)
    _check_linear(name, x, cuts, axis)


@pytest.mark.parametrize("dtype", [onp.float32, onp.complex128])
def test_split_overlap_cotangent_dtype(dtype):
    x = onp.linspace(0.3, 1.2, 10).reshape(2, 5).astype(dtype)
    if onp.iscomplexobj(x):
        x += 0.3j
    _check_linear("array_split", x, [-1, 1], -1, finite_difference=False)


@pytest.mark.parametrize("name, shape, axis", FAMILIES)
def test_split_overlap_nonlinear_second_derivative(name, shape, axis):
    x = onp.linspace(0.3, 1.2, onp.prod(shape)).reshape(shape)
    cuts = [-1, 1]
    parts = _split(onp, name, x, cuts, axis)
    labels = _split(onp, name, onp.arange(x.size).reshape(x.shape), cuts, axis)
    weights = _weights(parts, x.dtype)
    tangent = onp.linspace(-0.4, 0.7, x.size).reshape(x.shape)

    def objective(a):
        return sum(np.sum(w * p**3 / 3.0) for w, p in zip(weights, _split(np, name, a, cuts, axis)))

    expected_gradient = _scatter(x, labels, [w * p**2 for w, p in zip(weights, parts)])
    expected_hvp = _scatter(
        x,
        labels,
        [2.0 * w * p * t for w, p, t in zip(weights, parts, _split(onp, name, tangent, cuts, axis))],
    )
    numpy_objective = lambda a: sum(
        onp.sum(w * p**3 / 3.0) for w, p in zip(weights, _split(onp, name, a, cuts, axis))
    )
    _assert_array(expected_gradient, _finite_gradient(numpy_objective, x))
    _assert_array(grad(objective)(x), expected_gradient)

    def numpy_gradient(a):
        return _scatter(x, labels, [w * p**2 for w, p in zip(weights, _split(onp, name, a, cuts, axis))])

    finite_hvp = (numpy_gradient(x + 1e-5 * tangent) - numpy_gradient(x - 1e-5 * tangent)) / 2e-5
    _assert_array(expected_hvp, finite_hvp)
    reverse_hvp = grad(lambda a: np.sum(grad(objective)(a) * tangent))(x)
    gradient, forward_hvp = make_jvp(grad(objective))(x)(tangent)
    _assert_array(gradient, expected_gradient)
    _assert_array(reverse_hvp, expected_hvp)
    _assert_array(forward_hvp, expected_hvp)


@pytest.mark.parametrize("name, cuts, axis", [("split", 2, 1), ("array_split", 3, 2)])
def test_split_primal_errors_unchanged(name, cuts, axis):
    x = onp.ones((2, 5))
    for module in (onp, np):
        with pytest.raises((ValueError, IndexError)):
            _split(module, name, x, cuts, axis)
