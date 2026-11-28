import numpy as np
import pytest

import autograd.numpy as anp
from autograd import grad
from autograd.test_util import check_grads


@pytest.mark.parametrize(
    ("shape", "n", "axis"),
    [
        ((0,), 0, -1),
        ((0,), 1, -1),
        ((1,), 0, -1),
        ((1,), 1, -1),
        ((1,), 2, -1),
        ((2,), 0, -1),
        ((2,), 1, -1),
        ((2,), 3, -1),
        ((3,), 0, -1),
        ((3,), 1, -1),
        ((3,), 4, -1),
        ((4,), 0, -1),
        ((4,), 1, -1),
        ((4,), 5, -1),
        ((2, 3), 2, -1),
        ((2, 3), 3, -1),
        ((2, 3), 5, -1),
        ((3, 2), 1, 0),
        ((3, 2), 3, 0),
        ((3, 2), 5, 0),
        ((0, 2), 1, 0),
        ((2, 0), 1, -1),
        ((0, 2), 1, -1),
        ((2, 0), 1, 0),
    ],
)
def test_diff_vjp_empty_output(shape, n, axis):
    x = np.linspace(0.2, 1.4, np.prod(shape)).reshape(shape)
    expected_output = np.diff(x, n=n, axis=axis)
    np.testing.assert_array_equal(anp.diff(x, n=n, axis=axis), expected_output)
    weights = np.arange(1, expected_output.size + 1).reshape(expected_output.shape)

    def objective(z, numpy_module):
        return numpy_module.sum(numpy_module.diff(z, n=n, axis=axis) * weights)

    expected = np.empty(shape)
    for index in np.ndindex(shape):
        basis = np.zeros(shape)
        basis[index] = 1
        expected[index] = objective(basis, np)
    actual = grad(lambda z: objective(z, anp))(x)
    assert actual.shape == x.shape
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_diff_empty_output_preserves_gradient_dtype(dtype):
    x = np.arange(6, dtype=dtype).reshape(2, 3)
    actual = grad(lambda z: anp.real(anp.sum(anp.diff(z, n=4, axis=1))))(x)
    assert actual.dtype == x.dtype
    np.testing.assert_array_equal(actual, np.zeros_like(x))


@pytest.mark.parametrize(
    ("shape", "n", "axis"),
    [
        ((2,), 3, -1),
        ((3,), 4, -1),
        ((2, 3), 5, -1),
        ((3, 2), 5, 0),
        ((2, 3), 1, -1),
        ((2, 0), 1, 0),
    ],
)
def test_diff_empty_output_higher_order(shape, n, axis):
    x = np.linspace(0.2, 1.4, np.prod(shape)).reshape(shape)

    def objective(z):
        return anp.sum(anp.sin(anp.diff(z, n=n, axis=axis)) ** 2)

    check_grads(objective, modes=["fwd", "rev"], order=2)(x)
