import numpy as onp
import pytest
from numpy.testing import assert_allclose

import autograd.numpy as np
from autograd import grad, make_jvp
from autograd.test_util import check_grads

REDUCTIONS = ["sum", "mean", "prod", "var", "std"]


def _check_derivatives(positional, keyword, expected, x, second_order=True):
    assert_allclose(positional(x), expected(x))
    tangent = onp.linspace(0.2, 0.8, x.size).reshape(x.shape)
    actual_value, actual_jvp = make_jvp(positional)(x)(tangent)
    expected_value, expected_jvp = make_jvp(keyword)(x)(tangent)
    assert onp.shape(actual_value) == onp.shape(expected_value)
    assert onp.shape(actual_jvp) == onp.shape(expected_jvp)
    assert_allclose(actual_jvp, expected_jvp)
    weights = onp.linspace(0.7, 1.3, onp.size(expected_value)).reshape(onp.shape(expected_value))

    def objective(a):
        return np.sum(weights * positional(a) ** 2)

    def keyword_objective(a):
        return np.sum(weights * keyword(a) ** 2)

    assert_allclose(grad(objective)(x), grad(keyword_objective)(x))
    if second_order:
        check_grads(objective, modes=["fwd", "rev"], order=2)(x)


@pytest.mark.parametrize("name", REDUCTIONS)
@pytest.mark.parametrize("axis", [None, 1])
def test_reduction_positional_dtype(name, axis):
    x = onp.linspace(0.7, 1.4, 12).reshape(3, 4)[:, ::2]
    reduction = getattr(np, name)
    numpy_reduction = getattr(onp, name)

    def keyword(a):
        return reduction(a, axis=axis, dtype=onp.float64)

    _check_derivatives(
        lambda a: reduction(a, axis, onp.float64),
        keyword,
        lambda a: numpy_reduction(a, axis=axis, dtype=onp.float64),
        x,
    )
    if axis is None:
        _check_derivatives(reduction, lambda a: reduction(a), numpy_reduction, x)


@pytest.mark.parametrize("name", REDUCTIONS)
@pytest.mark.parametrize("axis, keepdims, ddof", [(None, False, 0), (1, True, 1), (-1, False, 0)])
@pytest.mark.parametrize("style", ["full", "mixed"])
def test_reduction_positional_options(name, axis, keepdims, ddof, style):
    x = onp.linspace(0.7, 1.4, 12).reshape(3, 4)[:, ::2]
    reduction = getattr(np, name)
    numpy_reduction = getattr(onp, name)
    kwargs = {"axis": axis, "dtype": onp.float64, "out": None, "keepdims": keepdims}
    if name in ("var", "std"):
        kwargs["ddof"] = ddof

    def positional(a):
        if name in ("var", "std"):
            if style == "full":
                return reduction(a, axis, onp.float64, None, ddof, keepdims)
            return reduction(a, axis, onp.float64, None, ddof, keepdims=keepdims)
        if style == "full":
            return reduction(a, axis, onp.float64, None, keepdims)
        return reduction(a, axis, onp.float64, keepdims=keepdims)

    _check_derivatives(
        positional,
        lambda a: reduction(a, **kwargs),
        lambda a: numpy_reduction(a, **kwargs),
        x,
    )


@pytest.mark.parametrize("name", REDUCTIONS)
def test_scalar_reduction_positional_options(name):
    x = onp.array(1.2)
    reduction = getattr(np, name)
    numpy_reduction = getattr(onp, name)
    kwargs = {"axis": None, "dtype": onp.float64, "out": None, "keepdims": False}
    if name in ("var", "std"):
        kwargs["ddof"] = 0

    def positional(a):
        if name in ("var", "std"):
            return reduction(a, None, onp.float64, None, 0, False)
        return reduction(a, None, onp.float64, None, False)

    _check_derivatives(
        positional,
        lambda a: reduction(a, **kwargs),
        lambda a: numpy_reduction(a, **kwargs),
        x,
        second_order=False,
    )
