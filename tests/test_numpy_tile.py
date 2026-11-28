import numpy as onp
import pytest

import autograd.numpy as np
from autograd import grad, make_jvp
from autograd.test_util import check_grads

REPETITION_CASES = [
    pytest.param((2, 3), 2, id="2d-scalar-short"),
    pytest.param((2, 3), (2,), id="2d-tuple-short"),
    pytest.param((2, 3), [2], id="2d-list-short"),
    pytest.param((2, 3), onp.array([2]), id="2d-array-short"),
    pytest.param((2, 3), onp.int64(2), id="numpy-integer"),
    pytest.param((2, 3), onp.array(2), id="zero-dimensional-reps"),
    pytest.param((2, 3), True, id="python-true"),
    pytest.param((2, 3), False, id="python-false"),
    pytest.param((2, 3), onp.bool_(True), id="numpy-true"),
    pytest.param((2, 3), onp.bool_(False), id="numpy-false"),
    pytest.param((2, 3), 1.0, id="python-float-one"),
    pytest.param((2, 3), (1.0, 1.0), id="tuple-float-ones"),
    pytest.param((2, 3), onp.float64(1.0), id="numpy-float-one"),
    pytest.param((3, 2), 2, id="short-nondivisible-first-axis"),
    pytest.param((2, 2, 3), 2, id="3d-scalar-short"),
    pytest.param((2, 3, 2), (2, 3), id="3d-tuple-short"),
    pytest.param((2, 3, 2), [2, 3], id="3d-list-short"),
    pytest.param((2, 3, 2), onp.array([2, 3]), id="3d-array-short"),
    pytest.param((2, 3), (1, 2), id="matching-ones"),
    pytest.param((2, 3), (2, 3), id="matching"),
    pytest.param((2, 3), [1, 2], id="matching-list"),
    pytest.param((2, 3), (onp.int64(1), onp.int64(2)), id="matching-numpy-integer"),
    pytest.param((2, 3), (2, 1, 3), id="longer"),
    pytest.param((2, 3), [2, 1, 3], id="longer-list"),
    pytest.param((2, 3, 2), (1, 2, 3), id="3d-matching"),
    pytest.param((2, 3, 2), (1, 1), id="short-all-ones"),
    pytest.param((3,), 2, id="1d-scalar"),
    pytest.param((3,), (2,), id="1d-tuple"),
    pytest.param((), 2, id="scalar-input"),
    pytest.param((), (), id="scalar-input-empty-reps"),
    pytest.param((2, 3), (), id="empty-tuple"),
    pytest.param((2, 3), [], id="empty-list"),
    pytest.param((2, 3), onp.array([], dtype=int), id="empty-array"),
    pytest.param((2, 3), 0, id="scalar-zero"),
    pytest.param((2, 3), (1, 0), id="matching-zero"),
    pytest.param((2, 3), (0, 2), id="leading-zero"),
    pytest.param((2, 3), onp.array(0), id="zero-dimensional-zero"),
    pytest.param((0, 3), (1, 2), id="empty-input"),
]


def tile_gradient_oracle(shape, reps, weights):
    # Scatter through the observed forward replication, independently of the VJP.
    labels = onp.arange(onp.prod(shape, dtype=int)).reshape(shape)
    result = onp.zeros(labels.size, dtype=weights.dtype)
    onp.add.at(result, onp.tile(labels, reps).ravel(), weights.ravel())
    return result.reshape(shape)


@pytest.mark.parametrize("shape,reps", REPETITION_CASES)
def test_tile_repetition_forms(shape, reps):
    x = onp.arange(onp.prod(shape, dtype=int), dtype=float).reshape(shape) / 7
    expected = onp.tile(x, reps)
    weights = onp.arange(1.0, expected.size + 1).reshape(expected.shape)
    tangent = onp.full_like(x, 0.75)

    def fun(z):
        return np.tile(z, reps)

    value, jvp = make_jvp(fun)(x)(tangent)
    assert onp.array_equal(value, expected)
    assert onp.array_equal(jvp, onp.tile(tangent, reps))
    expected_gradient = tile_gradient_oracle(shape, reps, weights)
    assert onp.array_equal(grad(lambda z: np.sum(fun(z) * weights))(x), expected_gradient)
    check_grads(lambda z: np.sin(fun(z)), modes=["fwd", "rev"], order=2)(x)


@pytest.mark.parametrize("dtype", [onp.float32, onp.float64, onp.complex64, onp.complex128])
@pytest.mark.parametrize("shape,reps", [((2, 3), 2), ((2, 3), 0), ((0, 3), (1, 2)), ((), (2, 3))])
def test_tile_gradient_dtype_and_layout(shape, reps, dtype):
    x = onp.arange(onp.prod(shape, dtype=int), dtype=float).reshape(shape).astype(dtype)
    if onp.issubdtype(dtype, onp.complexfloating):
        x = x + (x + 1) * dtype(0.25j)
    if x.ndim == 2:
        x = onp.asfortranarray(x)
    expected = onp.tile(x, reps)
    weights = onp.arange(1.0, expected.size + 1).reshape(expected.shape).astype(dtype)
    if onp.issubdtype(dtype, onp.complexfloating):
        weights += dtype(0.5j)

    actual = grad(lambda z: np.real(np.sum(np.tile(z, reps) * weights)))(x)
    expected_gradient = tile_gradient_oracle(shape, reps, weights)
    assert actual.dtype == x.dtype
    onp.testing.assert_allclose(actual, expected_gradient, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("reps", [-1, (1, -1), 1.5, onp.array([[2, 3], [1, 2]])])
def test_tile_invalid_repetitions(reps):
    x = onp.arange(6.0).reshape(2, 3)
    try:
        onp.tile(x, reps)
    except (TypeError, ValueError) as error:
        expected_exception = type(error)
    else:
        raise AssertionError("Expected NumPy to reject invalid repetitions")
    with pytest.raises(expected_exception):
        np.tile(x, reps)
    with pytest.raises(expected_exception):
        grad(lambda z: np.sum(np.tile(z, reps)))(x)
