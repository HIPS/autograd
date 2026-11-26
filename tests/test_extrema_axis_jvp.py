import numpy as np
import pytest

import autograd.numpy as anp
from autograd import make_jvp
from autograd.test_util import check_grads


@pytest.mark.parametrize("function", [anp.max, anp.min])
@pytest.mark.parametrize("axis", [(-2, -1), (-1, -2), (0, -1), (1, 2), np.int64(-1), (), None])
@pytest.mark.parametrize("keepdims", [False, True])
def test_extrema_tangents_restore_all_reduced_axes(function, axis, keepdims):
    x = np.arange(24.0).reshape(2, 3, 4)
    tangent = np.cos(x)
    _, actual = make_jvp(lambda value: function(value, axis=axis, keepdims=keepdims))(x)(tangent)
    reducer = np.max if function is anp.max else np.min
    winners = x == reducer(x, axis=axis, keepdims=True)
    expected = np.sum(tangent * winners, axis=axis, keepdims=keepdims) / np.sum(
        winners, axis=axis, keepdims=keepdims
    )
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("function", [anp.max, anp.min])
def test_extrema_negative_axis_tuples_keep_tied_tangent_average(function):
    x = np.ones((2, 3, 4))
    tangent = np.arange(24.0).reshape(x.shape)
    _, actual = make_jvp(lambda value: function(value, axis=(-2, -1)))(x)(tangent)
    np.testing.assert_allclose(actual, np.mean(tangent, axis=(1, 2)))


@pytest.mark.parametrize("function", [anp.max, anp.min])
def test_extrema_negative_axes_support_higher_derivatives(function):
    x = np.arange(24.0).reshape(2, 3, 4) / 10 + 1
    check_grads(lambda value: anp.sum(function(value, axis=(-2, -1)) ** 3), modes=["fwd", "rev"], order=2)(x)
