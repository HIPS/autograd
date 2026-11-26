import numpy as np
import pytest

import autograd.numpy as anp
from autograd import grad, hessian, jacobian


@pytest.mark.parametrize("input_shape", [(), (3,), (2, 3)])
@pytest.mark.parametrize("output_shape", [(0,), (2, 0, 3)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64])
def test_empty_output_jacobian_keeps_shape_and_input_dtype(input_shape, output_shape, dtype):
    x = np.ones(input_shape, dtype=dtype)
    function = lambda value: anp.reshape(anp.ravel(anp.real(value))[:0], output_shape)
    actual = jacobian(function)(x)
    assert actual.shape == output_shape + input_shape
    assert actual.dtype == x.dtype
    assert actual.size == 0


@pytest.mark.parametrize("shape", [(0,), (2, 0, 3)])
def test_jacobian_of_empty_identity(shape):
    x = np.empty(shape)
    actual = jacobian(lambda value: value)(x)
    assert actual.shape == shape + shape
    assert actual.size == 0


@pytest.mark.parametrize("shape", [(0,), (2, 0, 3)])
def test_hessian_of_empty_parameter_array(shape):
    x = np.empty(shape)
    actual = hessian(lambda value: anp.sum(value**2))(x)
    assert actual.shape == shape + shape
    assert actual.size == 0


def test_empty_jacobian_supports_higher_derivatives():
    x = np.ones((2, 3))
    function = lambda value: anp.sin(value)[:, :0]
    actual = jacobian(jacobian(function))(x)
    assert actual.shape == (2, 0, 2, 3, 2, 3)
    np.testing.assert_array_equal(
        grad(lambda value: anp.sum(jacobian(function)(value)))(x), np.zeros_like(x)
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_nonempty_jacobian_control(dtype):
    x = np.arange(3, dtype=dtype)
    np.testing.assert_allclose(jacobian(anp.sin)(x), np.diag(np.cos(x)))
