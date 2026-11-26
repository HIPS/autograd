import numpy as onp
import pytest
from scipy import special as scipy_special

import autograd.numpy as np
from autograd import grad
from autograd.scipy.special import ive
from autograd.test_util import check_grads


@pytest.mark.parametrize("order,expected", [(-1, 0.5), (1, 0.5), (2, 0.0), (3, 0.0)])
def test_scaled_bessel_gradient_at_zero(order, expected):
    f = lambda x: ive(order, x)
    with onp.errstate(divide="raise", invalid="raise"):
        onp.testing.assert_allclose(grad(f)(0.0), expected)
    check_grads(f, modes=["rev"], order=1)(0.0)


@pytest.mark.parametrize("order", [-2, -1, 1, 2, 3])
def test_scaled_bessel_gradient_in_array_containing_zero(order):
    x = np.array([-500.0, -2.0, 0.0, 1e-12, 1.3, 500.0])
    weights = np.arange(1.0, len(x) + 1)
    expected = np.exp(-np.abs(x)) * scipy_special.ivp(order, x) - np.sign(x) * scipy_special.ive(order, x)
    with onp.errstate(divide="raise", invalid="raise"):
        actual = grad(lambda x: np.sum(weights * ive(order, x)))(x)
    onp.testing.assert_allclose(actual, weights * expected, rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("order", [0, 0.5, 1, 2])
def test_scaled_bessel_nonzero_higher_derivatives(order):
    check_grads(lambda x: ive(order, x), modes=["rev"], order=2)(np.array([0.2, 1.0, 5.0]))


def test_scaled_bessel_second_derivative_at_zero():
    onp.testing.assert_allclose(grad(grad(lambda x: ive(2, x)))(0.0), 0.25)
