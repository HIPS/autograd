from contextlib import nullcontext

import numpy as np
import pytest

import autograd.numpy as anp
from autograd import grad, jacobian, make_jvp, make_vjp
from autograd.test_util import check_grads


def _copy_adjoint(value, result, cotangent):
    labels = np.full(result.shape, np.arange(np.size(value)).reshape(np.shape(value)), dtype=np.int64)
    if not np.iscomplexobj(value) or not np.iscomplexobj(result):
        cotangent = np.real(cotangent)
    expected = np.zeros(np.shape(value), dtype=np.asarray(value).dtype)
    for label, coefficient in zip(labels.flat, np.asarray(cotangent).flat):
        expected.flat[label] += coefficient
    return expected


def _assert_full_derivatives(value, shape, dtype=None):
    casts_complex_to_real = (
        np.iscomplexobj(value) and dtype is not None and not np.issubdtype(dtype, np.complexfloating)
    )
    with pytest.warns(np.exceptions.ComplexWarning) if casts_complex_to_real else nullcontext():
        expected_result = np.full(shape, value, dtype=dtype)
        actual_result = anp.full(shape, value, dtype=dtype)
        np.testing.assert_array_equal(actual_result, expected_result)
        weights = (1 + np.arange(expected_result.size).reshape(shape) * 0.03).astype(
            expected_result.real.dtype
        )
        cotangent = weights.astype(expected_result.dtype)
        if np.iscomplexobj(expected_result):
            cotangent += np.asarray(0.3j, dtype=expected_result.dtype) * weights

        pullback, _ = make_vjp(lambda z: anp.full(shape, z, dtype=dtype))(value)
        adjoint = pullback(cotangent)

        def objective(z):
            result = anp.full(shape, z, dtype=dtype)
            return anp.real(anp.sum(weights * result * result))

        actual_gradient = grad(objective)(value)
        tangent = (0.19 + np.arange(np.size(value)).reshape(np.shape(value)) * 0.02).astype(
            np.asarray(value).dtype
        )
        if np.iscomplexobj(value):
            tangent += np.asarray(0.07j, dtype=np.asarray(value).dtype)
        if isinstance(value, float):
            tangent = float(tangent)
        _, actual_jvp = make_jvp(lambda z: anp.full(shape, z, dtype=dtype))(value)(tangent)
        expected_jvp = np.full(shape, tangent, dtype=dtype)
        _, actual_hvp = make_jvp(grad(objective))(value)(tangent)
        low_precision = np.asarray(value).dtype in (np.dtype("float32"), np.dtype("complex64")) or (
            expected_result.dtype in (np.dtype("float32"), np.dtype("complex64"))
        )
        tolerance = 2e-5 if low_precision else 1e-12
        for actual, expected in (
            (adjoint, _copy_adjoint(value, expected_result, cotangent)),
            (actual_gradient, _copy_adjoint(value, expected_result, 2 * weights * expected_result)),
            (actual_hvp, _copy_adjoint(value, expected_result, 2 * weights * expected_jvp)),
        ):
            assert np.shape(actual) == np.shape(value)
            assert np.iscomplexobj(actual) == np.iscomplexobj(value)
            np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
        np.testing.assert_array_equal(actual_jvp, expected_jvp)


@pytest.mark.parametrize(
    "fill_shape,shape",
    [
        ((3,), (2, 3)),
        ((1, 3), (4, 3)),
        ((2, 1), (2, 4)),
        ((1, 2, 1), (3, 2, 4)),
        ((2, 3), (2, 3)),
        ((1,), (2, 3)),
        ((1, 3), (3,)),
        ((1, 1, 3), (2, 3)),
        ((1, 1), ()),
        ((3,), (0, 3)),
        ((0,), (2, 0)),
        ((1, 3), (0, 3)),
        ((0, 1), (0, 4)),
        ((), (2, 3)),
        ((), ()),
    ],
)
def test_full_fill_shape_and_broadcast_derivatives(fill_shape, shape):
    value = np.linspace(0.2, 0.8, int(np.prod(fill_shape))).reshape(fill_shape)
    _assert_full_derivatives(value, shape)


@pytest.mark.parametrize(
    "fill_shape,fill_dtype,dtype",
    [
        ((3,), np.float32, None),
        ((3,), np.float64, None),
        ((3,), np.complex64, None),
        ((3,), np.complex128, None),
        ((3,), np.float64, np.float32),
        ((3,), np.float32, np.float64),
        ((3,), np.float64, np.complex128),
        ((3,), np.complex128, np.float64),
        ((3,), np.complex64, np.float32),
        ((3,), np.complex128, np.complex64),
        ((), np.complex128, np.float64),
    ],
)
def test_full_real_complex_fill_derivatives(fill_shape, fill_dtype, dtype):
    value = np.linspace(0.2, 0.8, int(np.prod(fill_shape))).reshape(fill_shape).astype(fill_dtype)
    if np.issubdtype(fill_dtype, np.complexfloating):
        value += np.asarray(0.17j, dtype=fill_dtype)
    _assert_full_derivatives(value, (2, 3), dtype)


@pytest.mark.parametrize("fill_shape", [(3,), (1, 3)])
def test_full_noncontiguous_fill(fill_shape):
    value = np.linspace(0.2, 0.8, 6).reshape(fill_shape[:-1] + (6,))[..., ::-2]
    assert not value.flags.c_contiguous
    _assert_full_derivatives(value, (2, 3))


@pytest.mark.parametrize("shape", [(), (4,), (2, 3), (0, 3)])
def test_full_python_scalar_fill(shape):
    _assert_full_derivatives(0.4, shape)


@pytest.mark.parametrize("fill_shape", [(3,), (1, 3)])
def test_full_hessian_preserves_source_coordinate_axes(fill_shape):
    value = np.array([1.0, 2.0, 3.0]).reshape(fill_shape)
    actual = jacobian(grad(lambda z: anp.sum(anp.full((2, 3), z) ** 2)))(value)
    expected = 4 * np.eye(value.size).reshape(value.shape + value.shape)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("fill_shape,shape", [((3,), (2, 3)), ((1, 3), (4, 3)), ((1, 3), (3,))])
def test_full_second_order(fill_shape, shape):
    value = np.linspace(0.2, 0.8, int(np.prod(fill_shape))).reshape(fill_shape)
    check_grads(lambda z: anp.sum(anp.sin(anp.full(shape, z)) ** 2), modes=["fwd", "rev"], order=2)(value)


@pytest.mark.parametrize("fill_shape,shape", [((2,), (2, 3)), ((2, 3), (3,))])
def test_full_invalid_broadcast_preserves_numpy_error(fill_shape, shape):
    value = np.ones(fill_shape)
    with pytest.raises(ValueError) as expected:
        np.full(shape, value)
    with pytest.raises(type(expected.value)):
        anp.full(shape, value)
