import numpy as np
import pytest

import autograd.numpy as anp
from autograd import grad
from autograd.test_util import check_grads


def gradient_parts(x, axis, numpy_module):
    result = numpy_module.gradient(x, axis=axis)
    count = x.ndim if axis is None else (1 if isinstance(axis, int) else len(axis))
    return [result[i] for i in range(count)] if count > 1 else [result]


def weighted_gradient(x, axis, numpy_module):
    parts = gradient_parts(x, axis, numpy_module)
    weights = np.arange(1, x.size + 1).reshape(x.shape)
    return sum(numpy_module.sum(part * (weights + i * 0.25)) for i, part in enumerate(parts))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    ("shape", "axis"),
    [
        ((2,), None),
        ((3,), None),
        ((4,), None),
        ((5,), None),
        ((2,), -1),
        ((3,), (0,)),
        ((2, 3), None),
        ((3, 2), (1, 0)),
        ((2, 5), 0),
        ((5, 2), -1),
        ((3, 5), 0),
        ((5, 3), 1),
        ((4, 5), None),
        ((5, 4), (1, 0)),
        ((2, 3, 4), (0, 2)),
        ((2, 3, 4), (-1, -2)),
        ((4, 3, 2), (1, -1)),
    ],
)
def test_gradient_vjp_short_axes(shape, axis, dtype):
    x = np.linspace(0.2, 1.7, np.prod(shape), dtype=dtype).reshape(shape)
    np.testing.assert_allclose(weighted_gradient(x, axis, anp), weighted_gradient(x, axis, np))
    expected = np.empty(shape)
    for index in np.ndindex(shape):
        basis = np.zeros(shape)
        basis[index] = 1.0
        expected[index] = weighted_gradient(basis, axis, np)
    actual = grad(lambda z: weighted_gradient(z, axis, anp))(x)
    assert actual.shape == x.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("shape", "axis"),
    [((2,), None), ((3,), 0), ((2, 3), None), ((3, 2), (1, 0)), ((4, 5), None)],
)
def test_gradient_short_axes_higher_order(shape, axis):
    x = np.linspace(0.2, 1.7, np.prod(shape)).reshape(shape)

    def objective(z):
        parts = gradient_parts(z, axis, anp)
        return sum(anp.sum(anp.sin(part) ** 2) for part in parts)

    check_grads(objective, modes=["fwd", "rev"], order=2)(x)
