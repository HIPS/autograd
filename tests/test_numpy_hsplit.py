import numpy as np
import pytest

import autograd.numpy as anp
from autograd import grad, make_jvp, make_vjp
from autograd.test_util import check_grads


@pytest.mark.parametrize(
    "shape,indices",
    [
        ((6,), 3),
        ((6,), np.int64(3)),
        ((6,), [1, 4]),
        ((6,), np.array([1, 4])),
        ((6,), [0, 2, 6]),
        ((0,), 3),
        ((1,), 1),
        ((2, 6), 3),
        ((2, 6, 2), [1, 4]),
    ],
)
def test_hsplit_derivatives_preserve_input_shape(shape, indices):
    value = np.linspace(0.2, 1.1, int(np.prod(shape))).reshape(shape)
    weights = np.arange(1.0, value.size + 1).reshape(shape)
    expected_parts = np.hsplit(value, indices)
    weights_parts = np.hsplit(weights, indices)

    pullback, parts = make_vjp(lambda z: anp.hsplit(z, indices))(value)
    assert len(parts) == len(expected_parts)
    for actual, expected in zip(parts, expected_parts):
        np.testing.assert_array_equal(actual, expected)
    adjoint = pullback(weights_parts)
    assert adjoint.shape == value.shape
    np.testing.assert_array_equal(adjoint, weights)

    actual = grad(lambda z: sum(anp.sum(p * w) for p, w in zip(anp.hsplit(z, indices), weights_parts)))(
        value
    )
    assert actual.shape == value.shape
    np.testing.assert_array_equal(actual, weights)

    tangent = np.linspace(-0.4, 0.7, value.size).reshape(shape)
    _, actual_jvp = make_jvp(lambda z: anp.hsplit(z, indices))(value)(tangent)
    for actual, expected in zip(actual_jvp, np.hsplit(tangent, indices)):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_hsplit_vector_dtype(dtype):
    value = np.linspace(0.2, 1.1, 6).astype(dtype)
    weights = np.arange(1, 7, dtype=dtype)
    if np.issubdtype(dtype, np.complexfloating):
        value = value + np.asarray(0.3j, dtype=dtype)
        weights = weights + np.asarray(0.2j, dtype=dtype)
    pullback, _ = make_vjp(lambda z: anp.hsplit(z, [1, 4]))(value)
    actual = pullback(np.hsplit(weights, [1, 4]))
    assert actual.shape == value.shape and actual.dtype == value.dtype
    np.testing.assert_array_equal(actual, weights)


@pytest.mark.parametrize("selected", [0, 1, 2, 3])
def test_hsplit_vector_part_used_in_isolation(selected):
    value = np.linspace(0.2, 1.1, 6)
    labels = np.hsplit(np.arange(value.size), [0, 2, 6])
    expected = np.zeros_like(value)
    expected[labels[selected]] = 1.0
    actual = grad(lambda z: anp.sum(anp.hsplit(z, [0, 2, 6])[selected]))(value)
    assert actual.shape == value.shape
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("shape", [(6,), (2, 6)])
def test_hsplit_noncontiguous_input(shape):
    value = np.linspace(0.2, 1.1, int(np.prod(shape))).reshape(shape)[..., ::-1]
    weights = np.arange(1.0, value.size + 1).reshape(shape)
    actual = grad(lambda z: sum(anp.sum(p * w) for p, w in zip(anp.hsplit(z, 3), np.hsplit(weights, 3))))(
        value
    )
    assert actual.shape == value.shape
    np.testing.assert_array_equal(actual, weights)


@pytest.mark.parametrize("shape", [(6,), (2, 6)])
@pytest.mark.parametrize("indices", [3, [0, 2, 6]])
def test_hsplit_second_order(shape, indices):
    value = np.linspace(0.2, 0.8, int(np.prod(shape))).reshape(shape)

    def objective(z):
        return sum(anp.sum(anp.sin(part) ** 2) for part in anp.hsplit(z, indices))

    check_grads(objective, modes=["fwd", "rev"], order=2)(value)


@pytest.mark.parametrize("value,indices", [(np.array(1.0), 1), (np.ones(6), 0), (np.ones(6), 4)])
def test_hsplit_invalid_inputs_preserve_numpy_error(value, indices):
    with pytest.raises((ValueError, ZeroDivisionError)) as expected:
        np.hsplit(value, indices)
    with pytest.raises(type(expected.value)):
        anp.hsplit(value, indices)
