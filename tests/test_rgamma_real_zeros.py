import numpy as np
import pytest
import scipy.special

import autograd.numpy as anp
from autograd import elementwise_grad, grad
from autograd.scipy.special import rgamma
from autograd.test_util import check_grads


@pytest.mark.parametrize("n", [0, 1, 2, 5, 20])
def test_rgamma_derivative_at_real_zeros(n):
    expected = (-1.0) ** n * scipy.special.gamma(n + 1)
    np.testing.assert_allclose(grad(rgamma)(-float(n)), expected, rtol=2e-13)


@pytest.mark.parametrize("n", [0, 1, 2, 5, 20])
def test_rgamma_second_derivative_at_real_zeros(n):
    expected = -2 * (-1.0) ** n * scipy.special.gamma(n + 1) * scipy.special.digamma(n + 1)
    np.testing.assert_allclose(grad(grad(rgamma))(-float(n)), expected, rtol=2e-13)


@pytest.mark.parametrize("x", [-1e-310, 1e-310])
def test_rgamma_derivative_near_zero(x):
    np.testing.assert_allclose(grad(rgamma)(x), 1.0, rtol=1e-14)


def test_rgamma_mixed_array_derivatives_are_finite():
    x = np.array([-5.0, -2.0, -1.0, 0.0, 0.25, 0.75, 3.5, 180.0])
    expected = -scipy.special.rgamma(x[4:]) * scipy.special.digamma(x[4:])
    expected = np.concatenate(([-120.0, 2.0, -1.0, 1.0], expected))
    np.testing.assert_allclose(elementwise_grad(rgamma)(x), expected, rtol=2e-13)
    check_grads(lambda value: anp.sum(rgamma(value)), modes=["rev"], order=2)(x[:-1])


@pytest.mark.parametrize("x", [-3.2, -0.3, 0.2, 0.5, 2.7, 100.0])
def test_rgamma_ordinary_real_derivatives(x):
    expected = -scipy.special.rgamma(x) * scipy.special.digamma(x)
    np.testing.assert_allclose(grad(rgamma)(x), expected, rtol=2e-13)


@pytest.mark.parametrize("x", [1.2 + 2.3j, -2.0 + 0.3j, -1.0 + 300.0j])
def test_rgamma_preserves_complex_derivatives(x):
    expected = scipy.special.digamma(x) / -scipy.special.gamma(x)
    np.testing.assert_allclose(grad(lambda value: anp.real(rgamma(value)))(x), expected, rtol=2e-13)
