import inspect

import numpy as np
import pytest

import autograd.numpy as anp
from autograd import grad, hessian, make_jvp, make_vjp


def array_with_layout(layout, shape=(2, 3)):
    if layout == "strided":
        return np.arange(12.0).reshape(2, 6)[:, ::2]
    return np.array(np.arange(np.prod(shape), dtype=float).reshape(shape), order=layout)


def transform(x, operation, order="A", style="keyword"):
    shape = () if operation == "ravel" else ((3, 2),)
    if style == "method":
        return getattr(x, operation)(*shape, order=order)
    function = getattr(anp, operation)
    if order is None:
        return function(x, *shape)
    if style == "positional":
        return function(x, *shape, order)
    return function(x, *shape, order=order)


def reference(x, operation, order):
    return np.ravel(x, order=order) if operation == "ravel" else np.reshape(x, (3, 2), order=order)


@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("operation,cotangent_layout", [("reshape", "C"), ("reshape", "F"), ("ravel", "C")])
def test_auto_order_vjp_uses_primal_layout(layout, operation, cotangent_layout):
    x = array_with_layout(layout)
    effective_order = "F" if layout == "F" else "C"
    vjp, answer = make_vjp(lambda a: transform(a, operation))(x)
    cotangent = np.array(np.arange(answer.size, dtype=float).reshape(answer.shape), order=cotangent_layout)
    np.testing.assert_array_equal(answer, reference(x, operation, effective_order))
    np.testing.assert_array_equal(vjp(cotangent), np.reshape(cotangent, x.shape, order=effective_order))


@pytest.mark.parametrize("operation", ["reshape", "ravel"])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("tangent_layout", ["C", "F"])
def test_auto_order_jvp_uses_primal_layout(operation, layout, tangent_layout):
    x = array_with_layout(layout)
    tangent = array_with_layout(tangent_layout)
    effective_order = "F" if layout == "F" else "C"
    answer, derivative = make_jvp(lambda a: transform(a, operation))(x)(tangent)
    np.testing.assert_array_equal(answer, reference(x, operation, effective_order))
    np.testing.assert_array_equal(derivative, reference(tangent, operation, effective_order))


@pytest.mark.parametrize("operation", ["reshape", "ravel"])
@pytest.mark.parametrize("layout", ["C", "F"])
@pytest.mark.parametrize("order", [None, "C", "F"])
def test_explicit_and_default_order_controls(operation, layout, order):
    x = array_with_layout(layout)
    effective_order = "C" if order is None else order
    function = lambda a: transform(a, operation, order=order)
    vjp, answer = make_vjp(function)(x)
    cotangent = np.array(np.arange(answer.size, dtype=float).reshape(answer.shape), order="F")
    tangent = array_with_layout("F")
    _, derivative = make_jvp(function)(x)(tangent)
    np.testing.assert_array_equal(answer, reference(x, operation, effective_order))
    np.testing.assert_array_equal(vjp(cotangent), np.reshape(cotangent, x.shape, order=effective_order))
    np.testing.assert_array_equal(derivative, reference(tangent, operation, effective_order))


@pytest.mark.parametrize("operation", ["reshape", "ravel"])
@pytest.mark.parametrize("style", ["method", "positional"])
def test_auto_order_call_conventions(operation, style):
    x = array_with_layout("F")
    tangent = array_with_layout("C")
    function = lambda a: transform(a, operation, style=style)
    vjp, answer = make_vjp(function)(x)
    cotangent = np.arange(answer.size, dtype=float).reshape(answer.shape)
    _, derivative = make_jvp(function)(x)(tangent)
    np.testing.assert_array_equal(vjp(cotangent), np.reshape(cotangent, x.shape, order="F"))
    np.testing.assert_array_equal(derivative, reference(tangent, operation, "F"))


@pytest.mark.parametrize("operation", ["reshape", "ravel"])
def test_auto_order_second_derivatives_and_layout_preserving_differences(operation):
    x = array_with_layout("F") + 1.0
    weights = np.arange(1.0, 7.0).reshape(reference(x, operation, "F").shape)

    def loss(a):
        output = transform(a, operation)
        return anp.sum(weights * output**2)

    coefficients = np.reshape(weights, x.shape, order="F")
    np.testing.assert_allclose(grad(loss)(x), 2 * coefficients * x)
    expected_hessian = np.diag((2 * coefficients).ravel()).reshape(x.shape + x.shape)
    np.testing.assert_allclose(hessian(loss)(x), expected_hessian)
    numerical_gradient = np.empty_like(x)
    for index in np.ndindex(x.shape):
        plus, minus = x.copy(order="K"), x.copy(order="K")
        plus[index] += 1e-5
        minus[index] -= 1e-5
        assert plus.flags.f_contiguous and minus.flags.f_contiguous
        numerical_gradient[index] = (loss(plus) - loss(minus)) / 2e-5
    np.testing.assert_allclose(grad(loss)(x), numerical_gradient, rtol=1e-8, atol=1e-7)


@pytest.mark.parametrize("shape", [(6,), (1, 6), (6, 1), (1, 1, 6)])
@pytest.mark.parametrize("cotangent_layout", ["C", "F"])
def test_auto_order_both_contiguous_primal_uses_c_order(shape, cotangent_layout):
    x = np.arange(1.0, 7.0).reshape(shape)
    assert x.flags.c_contiguous and x.flags.f_contiguous
    weights = np.array(np.arange(1.0, 7.0).reshape(2, 3), order=cotangent_layout)
    function = lambda a: anp.reshape(a, (2, 3), order="A")
    vjp, answer = make_vjp(function)(x)
    tangent = np.arange(2.0, 8.0).reshape(shape)
    _, derivative = make_jvp(function)(x)(tangent)
    np.testing.assert_array_equal(answer, np.reshape(x, (2, 3), order="C"))
    np.testing.assert_array_equal(vjp(weights), weights.reshape(shape, order="C"))
    np.testing.assert_array_equal(derivative, tangent.reshape(2, 3, order="C"))

    loss = lambda a: anp.sum(function(a) * weights)
    numerical_gradient = np.empty_like(x)
    for index in np.ndindex(shape):
        plus, minus = x.copy(order="K"), x.copy(order="K")
        plus[index] += 1e-5
        minus[index] -= 1e-5
        numerical_gradient[index] = (loss(plus) - loss(minus)) / 2e-5
    np.testing.assert_allclose(grad(loss)(x), numerical_gradient, rtol=1e-8, atol=1e-7)


@pytest.mark.parametrize("layout", ["C", "F"])
@pytest.mark.parametrize("constant_tangent", [False, True])
@pytest.mark.parametrize("style", ["keyword", "positional"])
@pytest.mark.skipif(
    "copy" not in inspect.signature(np.reshape).parameters, reason="NumPy reshape has no copy keyword"
)
def test_auto_order_no_copy_primal_allows_different_tangent_layout(layout, constant_tangent, style):
    x = array_with_layout(layout)
    tangent = np.array(np.ones_like(x) if constant_tangent else x + 1, order="F" if layout == "C" else "C")
    if style == "keyword":
        function = lambda a: anp.reshape(a, (3, 2), order="A", copy=False)
    else:
        function = lambda a: anp.reshape(a, (3, 2), "A", copy=False)
    answer, derivative = make_jvp(function)(x)(tangent)
    assert np.shares_memory(answer, x)
    np.testing.assert_array_equal(derivative, np.reshape(tangent, (3, 2), order=layout))
    plus, minus = x.copy(order="K"), x.copy(order="K")
    plus += 1e-5 * tangent
    minus -= 1e-5 * tangent
    numerical_tangent = (function(plus) - function(minus)) / 2e-5
    np.testing.assert_allclose(derivative, numerical_tangent, rtol=1e-8, atol=1e-7)
