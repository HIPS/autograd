"""
Set of tests that are as explicit as possible, in case the test helpers like
autograd.test_util break and start letting everything pass
"""
import autograd.numpy as np
from autograd import grad, deriv

def test_grad():
    def fun(x): return (x + np.sin(x**2)) * x
    assert 3.190948746871 - 1e-6 < grad(fun)(1.3) < 3.190948746871 + 1e-6

def test_deriv():
    def fun(x): return (x + np.sin(x**2)) * x
    assert 3.190948746871 - 1e-6 < deriv(fun)(1.3) < 3.190948746871 + 1e-6
