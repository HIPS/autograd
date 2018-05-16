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
    """All that I'm asking of this test is that it doesn't error out."""
    x = cp.array([1, 2, 3])
    l2 = cp.linalg.norm(x)


@pytest.mark.integration
def test_argmin():
    """I just don't want this test to error out."""
    x = cp.arange(10)
    cp.argmin(x)

