import numpy as onp
import pytest

import autograd.numpy as np
from autograd import grad, hessian_vector_product, make_jvp, make_vjp
from autograd.test_util import check_grads


def trace_adjoint(shape, axis1, axis2, offset, weights):
    axis1, axis2 = axis1 % len(shape), axis2 % len(shape)
    remaining = [axis for axis in range(len(shape)) if axis not in (axis1, axis2)]
    result = onp.zeros(shape, dtype=onp.asarray(weights).dtype)
    for index in onp.ndindex(shape):
        if index[axis2] - index[axis1] == offset:
            result[index] = weights[tuple(index[axis] for axis in remaining)]
    return result


@pytest.mark.parametrize("style", ["keyword", "positional", "axis1-only", "axis2-only"])
def test_trace_explicit_default_axes(style):
    x = onp.linspace(0.2, 1.1, 6).reshape(2, 3)

    def fun(z):
        if style == "positional":
            return np.trace(z, 0, 0, 1)
        if style == "axis1-only":
            return np.trace(z, axis1=0)
        if style == "axis2-only":
            return np.trace(z, axis2=1)
        return np.trace(z, axis1=0, axis2=1)

    onp.testing.assert_allclose(grad(fun)(x), [[1, 0, 0], [0, 1, 0]])
    onp.testing.assert_allclose(fun(x), onp.trace(x))


@pytest.mark.parametrize(
    "shape,axis1,axis2,offset",
    [
        ((2, 3), 0, 1, 1),
        ((3, 2), 1, 0, -1),
        ((2, 3, 4), 0, 2, 1),
        ((2, 3, 4), 2, 1, -1),
        ((2, 3, 4), -1, -2, -1),
        ((2, 3, 4, 5), 3, 1, 1),
        ((2, 3, 4, 5), -1, -3, 1),
        ((2, 3, 4, 5), 1, 3, -1),
        ((2, 3, 4, 5), 3, 2, 0),
        ((2, 3, 4), 0, 2, 5),
        ((2, 3, 4), 0, 2, -5),
    ],
)
def test_trace_weighted_adjoint(shape, axis1, axis2, offset):
    x = onp.linspace(0.2, 1.1, onp.prod(shape)).reshape(shape)
    tangent = onp.linspace(-0.4, 0.7, x.size).reshape(shape)
    kwargs = {"offset": offset, "axis1": axis1, "axis2": axis2}
    trace = lambda z: np.trace(z, **kwargs)
    expected_value = onp.trace(x, **kwargs)
    weights = onp.arange(1.0, expected_value.size + 1).reshape(expected_value.shape)
    expected = trace_adjoint(shape, axis1, axis2, offset, weights)
    pullback, value = make_vjp(trace)(x)
    onp.testing.assert_array_equal(value, expected_value)
    onp.testing.assert_array_equal(pullback(weights), expected)
    loss = lambda z: np.sum(trace(z) * weights)
    onp.testing.assert_allclose(grad(loss)(x), expected)
    onp.testing.assert_allclose(make_jvp(trace)(x)(tangent)[1], onp.trace(tangent, **kwargs))
    step = 1e-5
    numerical = (loss(x + step * tangent) - loss(x - step * tangent)) / (2 * step)
    onp.testing.assert_allclose(numerical, onp.sum(expected * tangent), atol=1e-8)


@pytest.mark.parametrize("dtype", [onp.float32, onp.float64, onp.complex64, onp.complex128])
def test_trace_noncontiguous_real_and_complex(dtype):
    x = onp.linspace(0.2, 1.1, 24).reshape(4, 3, 2).transpose(2, 1, 0).astype(dtype)
    x = x[::-1, :, ::-1]
    if onp.iscomplexobj(x):
        x = x + 0.3j * x
    assert not x.flags.c_contiguous
    trace = lambda z: np.trace(z, offset=1, axis1=-1, axis2=0)
    value = onp.trace(x, offset=1, axis1=-1, axis2=0)
    weights = onp.arange(1.0, value.size + 1).reshape(value.shape).astype(dtype)
    if onp.iscomplexobj(x):
        weights = weights + 0.2j * weights
    expected = trace_adjoint(x.shape, -1, 0, 1, weights)
    pullback, actual = make_vjp(trace)(x)
    onp.testing.assert_array_equal(actual, value)
    onp.testing.assert_allclose(pullback(weights), expected)
    gradient = grad(lambda z: np.real(np.sum(trace(z) * weights)))(x)
    onp.testing.assert_allclose(gradient, expected)
    assert onp.iscomplexobj(gradient) == onp.iscomplexobj(x)
    tangent = onp.linspace(-0.4, 0.7, x.size).reshape(x.shape).astype(dtype)
    onp.testing.assert_array_equal(make_jvp(trace)(x)(tangent)[1], onp.trace(tangent, 1, -1, 0))


@pytest.mark.parametrize("shape,axes", [((0, 3), (0, 1)), ((2, 0, 3), (1, 2)), ((0, 2, 3), (1, 2))])
def test_trace_empty_dimensions(shape, axes):
    x = onp.empty(shape)
    trace = lambda z: np.trace(z, axis1=axes[0], axis2=axes[1])
    pullback, value = make_vjp(trace)(x)
    onp.testing.assert_array_equal(value, onp.trace(x, axis1=axes[0], axis2=axes[1]))
    gradient = pullback(onp.ones_like(value))
    assert gradient.shape == x.shape
    onp.testing.assert_array_equal(gradient, onp.zeros_like(x))


@pytest.mark.parametrize("shape,offset", [((2, 3), -1), ((2, 3, 4), 1), ((2, 3, 4, 5), -1), ((0, 3, 2), 0)])
def test_trace_default_axes_unchanged(shape, offset):
    x = onp.linspace(0.2, 1.1, onp.prod(shape)).reshape(shape)
    value = onp.trace(x, offset=offset)
    weights = onp.arange(1.0, value.size + 1).reshape(value.shape)
    expected = trace_adjoint(shape, 0, 1, offset, weights)
    pullback, actual = make_vjp(lambda z: np.trace(z, offset=offset))(x)
    onp.testing.assert_array_equal(actual, value)
    onp.testing.assert_array_equal(pullback(weights), expected)


@pytest.mark.parametrize("axes,offset", [((3, 1), 1), ((1, 3), -1), ((-1, -3), 1)])
def test_trace_hessian_vector_and_mixed_modes(axes, offset):
    shape = (2, 3, 4, 5)
    x = onp.linspace(0.2, 1.1, onp.prod(shape)).reshape(shape)
    tangent = onp.linspace(-0.4, 0.7, x.size).reshape(shape)
    kwargs = {"offset": offset, "axis1": axes[0], "axis2": axes[1]}
    value = onp.trace(x, **kwargs)
    weights = onp.arange(1.0, value.size + 1).reshape(value.shape)
    trace = lambda z: np.trace(z, **kwargs)
    loss = lambda z: 0.5 * np.sum(weights * trace(z) ** 2)
    expected_grad = trace_adjoint(shape, *axes, offset, weights * value)
    expected_hvp = trace_adjoint(shape, *axes, offset, weights * onp.trace(tangent, **kwargs))
    onp.testing.assert_allclose(grad(loss)(x), expected_grad)
    onp.testing.assert_allclose(hessian_vector_product(loss)(x, tangent), expected_hvp)
    onp.testing.assert_allclose(make_jvp(grad(loss))(x)(tangent)[1], expected_hvp)
    onp.testing.assert_allclose(grad(lambda z: make_jvp(loss)(z)(tangent)[1])(x), expected_hvp)
    check_grads(loss, modes=["fwd", "rev"], order=2)(x)


@pytest.mark.parametrize("axes,offset", [((-1, 0), 1), ((1, 2), -1)])
def test_trace_complex_second_order(axes, offset):
    shape = (2, 3, 4)
    x = onp.linspace(0.2, 1.1, onp.prod(shape)).reshape(shape) * (1 + 0.3j)
    tangent = onp.linspace(-0.4, 0.7, x.size).reshape(shape) * (1 - 0.2j)
    kwargs = {"offset": offset, "axis1": axes[0], "axis2": axes[1]}
    value = onp.trace(x, **kwargs)
    weights = onp.arange(1.0, value.size + 1).reshape(value.shape)
    trace = lambda z: np.trace(z, **kwargs)
    loss = lambda z: 0.5 * np.real(np.sum(weights * np.conj(trace(z)) * trace(z)))
    expected_grad = onp.conj(trace_adjoint(shape, *axes, offset, weights * value))
    expected_hvp = onp.conj(trace_adjoint(shape, *axes, offset, weights * onp.trace(tangent, **kwargs)))
    onp.testing.assert_allclose(grad(loss)(x), expected_grad)
    onp.testing.assert_allclose(make_jvp(grad(loss))(x)(tangent)[1], expected_hvp)
    onp.testing.assert_allclose(grad(lambda z: make_jvp(loss)(z)(tangent)[1])(x), expected_hvp)
    check_grads(loss, modes=["fwd", "rev"], order=2)(x)


@pytest.mark.parametrize("axes", [(1, 1), (-1, 2), (3, 1), (-4, 1), (0.5, 1)])
def test_trace_invalid_axes_match_numpy(axes):
    x = onp.arange(24.0).reshape(2, 3, 4)
    kwargs = {"axis1": axes[0], "axis2": axes[1]}
    with pytest.raises(Exception) as expected:
        onp.trace(x, **kwargs)
    for fun in (lambda z: np.trace(z, **kwargs), grad(lambda z: np.sum(np.trace(z, **kwargs)))):
        with pytest.raises(type(expected.value)) as actual:
            fun(x)
        assert str(actual.value) == str(expected.value)
