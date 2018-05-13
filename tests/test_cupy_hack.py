import autograd.cupy as cp
from autograd import elementwise_grad as egrad
import pytest


@pytest.mark.integration
def test_sin():
    a = cp.arange(10)

    def f(x):
        return cp.sin(x)

    df = egrad(f)

    df(a)

