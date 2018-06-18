import autograd.cupy as cp
from autograd import elementwise_grad as egrad
import pytest
from autograd import grad
import pdb
import numpy as np

@pytest.mark.integration
@pytest.mark.cupy
def test_sin():
    a = cp.arange(10)

    def f(x):
        return cp.sin(x)

    df = egrad(f)

    df(a)


@pytest.mark.higher_order
@pytest.mark.integration
@pytest.mark.cupy
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
@pytest.mark.cupy
def test_linalg_norm():
    """All that I'm asking of this test is that it doesn't error out."""
    x = cp.array([1, 2, 3])
    l2 = cp.linalg.norm(x)


@pytest.mark.integration
@pytest.mark.cupy
def test_argmin():
    """I just don't want this test to error out."""
    x = cp.arange(10)
    cp.argmin(x)


@pytest.mark.integration
@pytest.mark.gradient
@pytest.mark.cupy
def test_gradient_descent():
    # x = cp.array([[1,2,3,4,5,6,7,8]])
    # y = cp.array([[-1,-1,-1,-1,1,1,1,1]])
    x = cp.random.random(size=(10, 2))
    w_truth = cp.random.random(size=(2,1))
    y = cp.dot(x, w_truth)

    # def model(w, x):
    #     a = w[0] + cp.dot(w[1:], x)
    #     return a.T
    def model(w, x):
        return cp.dot(x, w)

    def loss(w, x, y):
        preds = model(w, x)
        loss_score = cp.mean(cp.power(preds.ravel() - y.ravel(), cp.array(2)))
        return loss_score

    w = cp.random.rand(2, 1)

    dloss = egrad(loss)

    a = dloss(w, x, y)
    for i in range(10000):
        w = w + -dloss(w, x, y) * 0.01
        print(w)
    print(w_truth)
    assert np.allclose(cp.asnumpy(w), cp.asnumpy(w_truth), atol=0.001)
