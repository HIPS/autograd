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


@pytest.mark.integration
def test_linalg_norm():
    x_gpu = cp.array([1, 2, 3])
    l2_gpu = cp.linalg.norm(x_gpu)
