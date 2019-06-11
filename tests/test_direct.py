"""
Set of tests that are as explicit as possible, in case the test helpers like
autograd.test_util break and start letting everything pass
"""
import numpy as onp
import autograd.numpy as np
from autograd import grad, deriv, holomorphic_grad
import pytest

def test_grad():
    def fun(x): return (x + np.sin(x**2)) * x
    assert 3.190948746871 - 1e-6 < grad(fun)(1.3) < 3.190948746871 + 1e-6

def test_deriv():
    def fun(x): return (x + np.sin(x**2)) * x
    assert 3.190948746871 - 1e-6 < deriv(fun)(1.3) < 3.190948746871 + 1e-6

def test_grad_complex_output():
    def fun(x): return x * (1.0 + 0.2j)
    with pytest.raises(TypeError):
        grad(fun)(1.0)

def test_holomorphic_grad():
    def fun(x): return x * (1.0 + 0.2j)
    g = holomorphic_grad(fun)(1.0 + 0.0j)
    assert 0.9999 < onp.real(g) < 1.0001
    assert 0.1999 < onp.imag(g) < 0.2001
