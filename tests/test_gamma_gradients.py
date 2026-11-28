import numpy as onp
import pytest

scipy = pytest.importorskip("scipy")
from scipy.stats import gamma as scipy_gamma

import autograd.numpy as np
from autograd import grad, hessian
from autograd.scipy import special, stats
from autograd.test_util import check_grads

CDF_CASES = [
    pytest.param(special.gammainc, 1, id="gammainc"),
    pytest.param(special.gammaincc, -1, id="gammaincc"),
    pytest.param(lambda a, x: stats.gamma.cdf(x, a), 1, id="gamma_cdf"),
]


@pytest.mark.parametrize("fun, sign", CDF_CASES)
@pytest.mark.parametrize("a", [200.0, 1000.0])
@pytest.mark.parametrize("factor", [0.5, 1.0, 2.0])
def test_gamma_cdf_large_shape_derivatives(fun, sign, a, factor):
    x = factor * a
    f = lambda x: fun(a, x)
    density = scipy_gamma.pdf(x, a)
    slope = (a - 1) / x - 1
    expected = (
        sign * density,
        sign * density * slope,
        sign * density * (slope**2 - (a - 1) / x**2),
    )
    # Tail densities are small but representable. An absolute tolerance would
    # incorrectly accept zero gradients for these cases.
    assert density > 0
    with onp.errstate(over="raise", invalid="raise", divide="raise"):
        for derivative in expected:
            f = grad(f)
            actual = f(x)
            assert onp.isfinite(actual)
            onp.testing.assert_allclose(actual, derivative, rtol=1e-10, atol=0)


@pytest.mark.parametrize("fun, sign", CDF_CASES)
@pytest.mark.parametrize(
    "a, expected",
    [(1.0, (1.0, -1.0, 1.0)), (2.0, (0.0, 1.0, -2.0)), (3.0, (0.0, 0.0, 1.0)), (200.0, (0.0, 0.0, 0.0))],
)
def test_gamma_cdf_zero_boundary_derivatives(fun, sign, a, expected):
    # Preserve the right-hand boundary convention. A centered finite difference
    # would cross the special functions' domain or the distribution's support.
    f = lambda x: fun(a, x)
    with onp.errstate(over="raise", invalid="raise", divide="raise"):
        for derivative in expected:
            f = grad(f)
            onp.testing.assert_allclose(f(0.0), sign * derivative, rtol=1e-14, atol=0)


@pytest.mark.parametrize("fun, sign", CDF_CASES)
def test_gamma_cdf_infinite_boundary_slope(fun, sign):
    with pytest.warns(RuntimeWarning, match="divide by zero"):
        actual = grad(lambda x: fun(0.5, x))(0.0)
    assert actual == sign * onp.inf


@pytest.mark.parametrize("fun, sign", CDF_CASES)
@pytest.mark.parametrize("a", [0.5, 1.0, 2.0, 200.0])
def test_gamma_cdf_positive_infinity(fun, sign, a):
    f = lambda x: fun(a, x)
    with onp.errstate(over="raise", invalid="raise", divide="raise"):
        for _ in range(2):
            f = grad(f)
            assert f(onp.inf) == 0


@pytest.mark.parametrize("a", [0.5, 1.0, 2.0, 200.0])
@pytest.mark.parametrize("x", [-1.0, -1000.0, -onp.inf])
def test_gamma_cdf_below_support(a, x):
    assert stats.gamma.cdf(x, a) == 0
    f = lambda x: stats.gamma.cdf(x, a)
    with onp.errstate(over="raise", invalid="raise", divide="raise"):
        for _ in range(2):
            f = grad(f)
            assert f(x) == 0


@pytest.mark.parametrize("fun", [special.gammainc, special.gammaincc])
@pytest.mark.parametrize("x", [1.0, 200.0, 1000.0])
def test_incomplete_gamma_zero_shape_extension(fun, x):
    with onp.errstate(over="raise", invalid="raise", divide="raise"):
        assert grad(lambda x: fun(0.0, x))(x) == 0


@pytest.mark.parametrize("fun, sign", CDF_CASES)
@pytest.mark.parametrize("broadcast", ["scalar", "row", "column"])
def test_gamma_cdf_broadcast_derivatives(fun, sign, broadcast):
    if broadcast == "scalar":
        a = onp.array([190.0, 200.0, 210.0])
        x = onp.array(200.0)
        weights = onp.array([-1.0, 2.0, 3.0])
        reduce = lambda value: onp.sum(value)
    elif broadcast == "row":
        a = onp.array([[200.0], [250.0]])
        x = onp.array([[200.0, 220.0, 240.0]])
        weights = onp.array([[1.0, -2.0, 3.0], [2.0, 1.0, -1.0]])
        reduce = lambda value: onp.sum(value, axis=0, keepdims=True)
    else:
        a = onp.array([[200.0, 220.0, 240.0]])
        x = onp.array([[200.0], [250.0]])
        weights = onp.array([[1.0, -2.0, 3.0], [2.0, 1.0, -1.0]])
        reduce = lambda value: onp.sum(value, axis=1, keepdims=True)
    f = lambda x: np.sum(weights * fun(a, x))
    density = scipy_gamma.pdf(x, a)
    expected = sign * reduce(weights * density)
    second = sign * reduce(weights * density * ((a - 1) / x - 1))
    expected_hessian = onp.diag(onp.ravel(second)).reshape(x.shape + x.shape)
    with onp.errstate(over="raise", invalid="raise", divide="raise"):
        actual = grad(f)(x)
        assert actual.shape == x.shape
        onp.testing.assert_allclose(actual, expected, rtol=1e-10, atol=0)
        onp.testing.assert_allclose(hessian(f)(x), expected_hessian, rtol=1e-10, atol=0)
    check_grads(f, modes=["rev"], order=2)(x)


@pytest.mark.parametrize("fun, sign", CDF_CASES)
def test_gamma_cdf_mixed_boundary_and_large_shape(fun, sign):
    a = onp.array([1.0, 200.0, 1000.0])
    x = onp.array([0.0, 200.0, 1000.0])
    f = lambda x: np.sum(fun(a, x))
    expected = sign * scipy_gamma.pdf(x, a)
    second = onp.array([-1.0, -scipy_gamma.pdf(200.0, 200.0) / 200, -scipy_gamma.pdf(1000.0, 1000.0) / 1000])
    with onp.errstate(over="raise", invalid="raise", divide="raise"):
        onp.testing.assert_allclose(grad(f)(x), expected, rtol=1e-10, atol=0)
        onp.testing.assert_allclose(hessian(f)(x), sign * onp.diag(second), rtol=1e-10, atol=0)


@pytest.mark.parametrize("fun, sign", CDF_CASES)
@pytest.mark.parametrize("a", [0.5, 1.0, 2.0, 20.0])
def test_gamma_cdf_interior_control(fun, sign, a):
    x = a + 0.3
    f = lambda x: fun(a, x)
    onp.testing.assert_allclose(grad(f)(x), sign * scipy_gamma.pdf(x, a), rtol=1e-13)
    check_grads(f, modes=["rev"], order=2)(x)


@pytest.mark.parametrize("fun, sign", CDF_CASES)
@pytest.mark.parametrize("a", [1.0, 2.0, 3.0])
@pytest.mark.parametrize("x", [1e-200, 1e-16, 1e-12, 0.5, 1.0 - 1e-8, 1.0, 1.0 + 1e-8])
def test_gamma_cdf_small_x_derivatives(fun, sign, a, x):
    # Closed forms avoid cancellation and division by x in the oracle.
    if a == 1:
        expected = (1.0, -1.0, 1.0)
    elif a == 2:
        expected = (x, 1 - x, x - 2)
    else:
        expected = (x**2 / 2, x - x**2 / 2, 1 - 2 * x + x**2 / 2)
    f = lambda x: fun(a, x)
    with onp.errstate(over="raise", invalid="raise", divide="raise"):
        for derivative in expected:
            f = grad(f)
            # Around x=1, the a=2 second derivative crosses zero. Allow roundoff
            # there, but keep a strictly relative check for the tiny-x cases.
            atol = onp.finfo(float).eps if x >= 0.5 else 0
            onp.testing.assert_allclose(f(x), sign * onp.exp(-x) * derivative, rtol=1e-13, atol=atol)


@pytest.mark.parametrize("fun, sign", CDF_CASES)
@pytest.mark.parametrize("x", [0.0, 0.5, 1.0, 2.0, 200.0])
def test_gamma_cdf_infinite_shape(fun, sign, x):
    f = lambda x: fun(onp.inf, x)
    with onp.errstate(over="raise", invalid="raise", divide="raise"):
        for _ in range(2):
            f = grad(f)
            assert f(x) == 0
