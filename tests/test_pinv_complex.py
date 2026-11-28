import numpy as onp
import pytest

import autograd.numpy as np
from autograd import grad, hessian
from autograd.test_util import check_grads


def adjoint(a):
    return np.conj(np.swapaxes(a, -1, -2))


def full_rank_pinv(a):
    if a.shape[-2] > a.shape[-1]:
        return np.matmul(np.linalg.inv(np.matmul(adjoint(a), a)), adjoint(a))
    if a.shape[-2] < a.shape[-1]:
        return np.matmul(adjoint(a), np.linalg.inv(np.matmul(a, adjoint(a))))
    return np.linalg.inv(a)


def matrix_and_weight(shape, batch, cotangent, complex_input=True):
    rng = onp.random.default_rng(721)
    m, n = shape
    a = onp.eye(m, n) * 2 + 0.2 * rng.standard_normal(batch + shape)
    if complex_input:
        a = a + 0.2j * rng.standard_normal(a.shape)
    real = rng.standard_normal(batch + (n, m))
    imag = rng.standard_normal(real.shape)
    weight = {"real": real, "imag": 1j * imag, "mixed": real + 1j * imag}[cotangent]
    return a, weight


@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("part", ["real", "imag"])
@pytest.mark.parametrize("t", [-0.7, 0.0, 1.0])
def test_pinv_complex_vector_closed_form(wide, part, t):
    def fun(t):
        a = np.array([[1 + 1j * t], [2.0]])
        if wide:
            a = np.transpose(a)
        return getattr(np, part)(np.linalg.pinv(a)[0, 0])

    denominator = 5 + t**2
    if part == "real":
        expected = (1 / denominator, -2 * t / denominator**2, (6 * t**2 - 10) / denominator**3)
    else:
        expected = (-t / denominator, (t**2 - 5) / denominator**2, 2 * t * (15 - t**2) / denominator**3)
    actual = (fun(t), grad(fun)(t), grad(grad(fun))(t))
    onp.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("shape", [(3, 2), (2, 3), (2, 2)])
@pytest.mark.parametrize("batch", [(), (2,), (2, 1)])
@pytest.mark.parametrize("cotangent", ["real", "imag", "mixed"])
def test_pinv_complex_full_rank_derivatives(shape, batch, cotangent):
    a, weight = matrix_and_weight(shape, batch, cotangent)
    coordinates = onp.stack([a.real, a.imag])

    def loss(pinv, coordinates):
        matrix = coordinates[0] + 1j * coordinates[1]
        return np.real(np.sum(weight * pinv(matrix)))

    fun = lambda coordinates: loss(np.linalg.pinv, coordinates)
    reference = lambda coordinates: loss(full_rank_pinv, coordinates)
    # Real coordinates cover each real and imaginary input direction, without
    # interpreting a complex Hessian as if the function were holomorphic.
    onp.testing.assert_allclose(fun(coordinates), reference(coordinates), rtol=1e-12, atol=1e-12)
    onp.testing.assert_allclose(grad(fun)(coordinates), grad(reference)(coordinates), rtol=1e-10, atol=1e-12)
    onp.testing.assert_allclose(
        hessian(fun)(coordinates), hessian(reference)(coordinates), rtol=1e-9, atol=1e-11
    )


@pytest.mark.parametrize("shape", [(3, 2), (2, 3), (2, 2)])
@pytest.mark.parametrize("batch", [(), (2,)])
def test_pinv_real_full_rank_control(shape, batch):
    a, weight = matrix_and_weight(shape, batch, "real", complex_input=False)
    fun = lambda a: np.sum(weight * np.linalg.pinv(a))
    reference = lambda a: np.sum(weight * full_rank_pinv(a))
    onp.testing.assert_allclose(grad(fun)(a), grad(reference)(a), rtol=1e-10, atol=1e-12)
    onp.testing.assert_allclose(hessian(fun)(a), hessian(reference)(a), rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("shape", [(3, 2), (2, 3)])
@pytest.mark.parametrize("batch", [(), (2,)])
def test_pinv_complex_gradcheck(shape, batch):
    a, _ = matrix_and_weight(shape, batch, "mixed")
    check_grads(np.linalg.pinv, modes=["rev"], order=2)(a)


@pytest.mark.parametrize("cotangent", ["real", "imag", "mixed"])
def test_pinv_complex_constant_rank(cotangent):
    u, _ = matrix_and_weight((3, 2), (), "mixed")
    v, weight = matrix_and_weight((2, 4), (), cotangent)
    rng = onp.random.default_rng(408)
    du = 0.1 * (rng.standard_normal(u.shape) + 1j * rng.standard_normal(u.shape))
    dv = 0.1 * (rng.standard_normal(v.shape) + 1j * rng.standard_normal(v.shape))
    weight = rng.standard_normal((4, 3))
    if cotangent == "imag":
        weight = 1j * weight
    elif cotangent == "mixed":
        weight = weight + 1j * rng.standard_normal(weight.shape)

    def loss(parameters, reference=False):
        left = u + parameters[0] * du
        right = v + parameters[1] * dv
        if reference:
            # U is full column rank and V full row rank, so (U V)^+ = V^+ U^+.
            inverse = np.matmul(full_rank_pinv(right), full_rank_pinv(left))
        else:
            inverse = np.linalg.pinv(np.matmul(left, right))
        return np.real(np.sum(weight * inverse))

    parameters = onp.array([0.2, -0.15])
    matrix = onp.matmul(u + parameters[0] * du, v + parameters[1] * dv)
    assert onp.linalg.matrix_rank(matrix) == 2
    reference = lambda parameters: loss(parameters, reference=True)
    onp.testing.assert_allclose(loss(parameters), reference(parameters), rtol=1e-12, atol=1e-12)
    onp.testing.assert_allclose(grad(loss)(parameters), grad(reference)(parameters), rtol=1e-10, atol=1e-12)
    onp.testing.assert_allclose(
        hessian(loss)(parameters), hessian(reference)(parameters), rtol=1e-9, atol=1e-11
    )
    # Perturb the factors, not arbitrary entries of the rank-deficient matrix.
    check_grads(loss, modes=["rev"], order=2)(parameters)
