import numpy as np
import pytest

import autograd.numpy as anp
from autograd import grad
from autograd.test_util import check_grads


def rotate(x, k, axes, call_style, numpy_module):
    if call_style == "omitted_axes":
        return numpy_module.rot90(x, k=k)
    if call_style == "positional_axes":
        return numpy_module.rot90(x, k, axes)
    return numpy_module.rot90(x, k=k, axes=axes)


def expected_rotation_vjp(shape, k, axes, call_style, weights):
    result = np.empty(shape)
    for index in np.ndindex(shape):
        basis = np.zeros(shape)
        basis[index] = 1
        result[index] = np.sum(rotate(basis, k, axes, call_style, np) * weights)
    return result


@pytest.mark.parametrize(
    ("shape", "k", "axes", "call_style"),
    [
        ((2, 3), -1, None, "omitted_axes"),
        ((2, 3), 0, None, "omitted_axes"),
        ((2, 3), 1, None, "omitted_axes"),
        ((2, 3), 2, None, "omitted_axes"),
        ((2, 3), 3, None, "omitted_axes"),
        ((2, 3), -1, (0, 1), "keyword_axes"),
        ((2, 3), 0, (0, 1), "keyword_axes"),
        ((2, 3), 1, (0, 1), "keyword_axes"),
        ((2, 3), 2, (0, 1), "keyword_axes"),
        ((2, 3), 3, (0, 1), "keyword_axes"),
        ((2, 3, 4), -1, (0, 2), "keyword_axes"),
        ((2, 3, 4), 1, (0, 2), "keyword_axes"),
        ((2, 3, 4), -1, (1, 2), "keyword_axes"),
        ((2, 3, 4), 1, (1, 2), "keyword_axes"),
        ((2, 3, 4), -1, (2, 0), "keyword_axes"),
        ((2, 3, 4), 1, (2, 0), "keyword_axes"),
        ((2, 3, 4), -1, (-1, 0), "keyword_axes"),
        ((2, 3, 4), 1, (-1, 0), "keyword_axes"),
        ((2, 3, 4), -1, (-2, -1), "keyword_axes"),
        ((2, 3, 4), 1, (-2, -1), "keyword_axes"),
        ((2, 3, 4), 1, None, "omitted_axes"),
        ((2, 3, 4), 2, None, "omitted_axes"),
        ((2, 3), -1, (0, 1), "positional_axes"),
        ((2, 3, 4), 1, (0, 2), "positional_axes"),
    ],
)
def test_rot90_axes_vjp(shape, k, axes, call_style):
    x = np.linspace(0.2, 1.7, np.prod(shape)).reshape(shape)
    output = rotate(x, k, axes, call_style, np)
    np.testing.assert_array_equal(rotate(x, k, axes, call_style, anp), output)
    weights = np.arange(1, output.size + 1).reshape(output.shape)
    actual = grad(lambda z: anp.sum(rotate(z, k, axes, call_style, anp) * weights))(x)
    assert actual.shape == x.shape
    np.testing.assert_array_equal(actual, expected_rotation_vjp(shape, k, axes, call_style, weights))


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_rot90_axes_gradient_dtype(dtype):
    x = np.arange(24, dtype=dtype).reshape(2, 3, 4)
    output = np.rot90(x, k=-1, axes=(2, 0))
    weights = np.arange(1, output.size + 1, dtype=dtype).reshape(output.shape)
    actual = grad(lambda z: anp.real(anp.sum(anp.rot90(z, k=-1, axes=(2, 0)) * weights)))(x)
    assert actual.dtype == x.dtype
    np.testing.assert_array_equal(
        actual, expected_rotation_vjp(x.shape, -1, (2, 0), "keyword_axes", weights.real)
    )


@pytest.mark.parametrize(
    ("shape", "k", "axes"),
    [
        ((2, 3), 1, None),
        ((2, 3), -1, (1, 0)),
        ((2, 3, 4), 0, (0, 2)),
        ((2, 3, 4), 2, (-1, -3)),
        ((2, 3, 4), 3, [-2, -1]),
    ],
)
def test_rot90_axes_higher_order(shape, k, axes):
    x = np.linspace(0.2, 1.7, np.prod(shape)).reshape(shape)
    call_style = "omitted_axes" if axes is None else "keyword_axes"
    check_grads(
        lambda z: anp.sum(anp.sin(rotate(z, k, axes, call_style, anp)) ** 2),
        modes=["fwd", "rev"],
        order=2,
    )(x)
