import autograd.cupy as cp
from autograd import elementwise_grad as egrad
import pytest
from autograd import grad
import pdb

@pytest.mark.integration
def test_sin():
    a = cp.arange(10)

    def f(x):
        return cp.sin(x)

    df = egrad(f)

    df(a)


@pytest.mark.higher_order
@pytest.mark.integration
def test_higher_order_derivatives():
    a = cp.arange(10)
    def f(x):
        return cp.sin(x)

    df = egrad(f)
    ddf = egrad(df)
    dddf = egrad(ddf)

    df(a)
    ddf(a)
    dddf(a)


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


@pytest.mark.integration
@pytest.mark.gradient
def test_gradient_descent():
    x = cp.array([[1,2,3,4,5,6,7,8]])
    y = cp.array([[-1,-1,-1,-1,1,1,1,1]])

    def model(w, x):
        a = w[0] + cp.dot(x.T,w[1:])
        return a.T

    def softmax(w):
        cost = cp.sum(cp.log(1 + cp.exp(-y*model(w, x))))
        return cost/float(cp.size(y))

    w = cp.random.rand(2,1)

    gradient = grad(softmax)

    a = gradient(w)
    print(a)
