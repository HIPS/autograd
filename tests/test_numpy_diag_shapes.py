import numpy as np
import pytest

import autograd.numpy as anp
from autograd import grad
from autograd.test_util import check_grads


def diag_basis_vjp(shape, k, weights):
    expected = np.empty(shape)
    for index in np.ndindex(shape):
        basis = np.zeros(shape)
        basis[index] = 1
        expected[index] = np.sum(np.diag(basis, k) * weights)
    return expected


@pytest.mark.parametrize("shape", [(2, 3), (3, 2), (2, 5), (2, 2), (3,), (0, 3), (3, 0), (0,)])
@pytest.mark.parametrize("k", [-2, -1, 0, 1, 2])
def test_diag_vjp_preserves_input_shape(shape, k):
    x = np.linspace(0.2, 1.7, np.prod(shape)).reshape(shape)
    output = np.diag(x, k)
    np.testing.assert_array_equal(anp.diag(x, k), output)
    weights = np.arange(1, output.size + 1).reshape(output.shape)
    actual = grad(lambda z: anp.sum(anp.diag(z, k) * weights))(x)
    assert actual.shape == x.shape
    np.testing.assert_array_equal(actual, diag_basis_vjp(shape, k, weights))


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("k", [-1, 0, 1])
def test_diag_matrix_gradient_dtype(dtype, k):
    x = np.arange(6, dtype=dtype).reshape(2, 3)
    weights = np.arange(1, np.diag(x, k).size + 1)
    actual = grad(lambda z: anp.real(anp.sum(anp.diag(z, k) * weights)))(x)
    assert actual.dtype == x.dtype
    np.testing.assert_array_equal(actual, diag_basis_vjp(x.shape, k, weights))


@pytest.mark.parametrize("shape", [(2, 2), (2, 3), (3,)])
@pytest.mark.parametrize("container", [list, tuple])
def test_diag_array_like_compatibility(shape, container):
    array = np.arange(np.prod(shape), dtype=float).reshape(shape)
    x = array.tolist() if container is list else tuple(array)
    actual = grad(lambda z: anp.sum(anp.diag(z)))(x)
    assert isinstance(actual, np.ndarray)
    np.testing.assert_array_equal(actual, diag_basis_vjp(shape, 0, np.ones_like(np.diag(array))))


@pytest.mark.parametrize(
    ("shape", "k"),
    [
        ((2, 3), -2),
        ((2, 3), -1),
        ((2, 3), 0),
        ((2, 3), 2),
        ((3, 2), -2),
        ((3, 2), 1),
        ((3,), -1),
        ((2, 2), 1),
    ],
)
def test_diag_higher_order(shape, k):
    x = np.linspace(0.2, 1.7, np.prod(shape)).reshape(shape)
    check_grads(lambda z: anp.sum(anp.sin(anp.diag(z, k)) ** 2), modes=["fwd", "rev"], order=2)(x)
