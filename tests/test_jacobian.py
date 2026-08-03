import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad, jacobian
from autograd.test_util import check_grads


def test_jacobian_against_grad():
    rng = npr.RandomState(42)
    fun = lambda x: np.sum(np.sin(x), axis=1, keepdims=True)
    A = rng.randn(1, 3)
    assert np.allclose(grad(fun)(A), jacobian(fun)(A))


def test_jacobian_scalar_to_vector():
    rng = npr.RandomState(42)
    fun = lambda x: np.array([x, x**2, x**3])
    val = rng.randn()
    assert np.allclose(jacobian(fun)(val), np.array([1.0, 2 * val, 3 * val**2]))


def test_jacobian_against_stacked_grads():
    rng = npr.RandomState(42)
    scalar_funs = [
        lambda x: np.sum(x**3),
        lambda x: np.prod(np.sin(x) + np.sin(x)),
        lambda x: grad(lambda y: np.exp(y) * np.tanh(x[0]))(x[1]),
    ]

    vector_fun = lambda x: np.array([f(x) for f in scalar_funs])

    x = rng.randn(5)
    jac = jacobian(vector_fun)(x)
    grads = [grad(f)(x) for f in scalar_funs]

    assert np.allclose(jac, np.vstack(grads))


def test_jacobian_higher_order():
    rng = npr.RandomState(42)
    fun = lambda x: np.sin(np.outer(x, x)) + np.cos(np.dot(x, x))

    assert jacobian(fun)(rng.randn(2)).shape == (2, 2, 2)
    assert jacobian(jacobian(fun))(rng.randn(2)).shape == (2, 2, 2, 2)
    # assert jacobian(jacobian(jacobian(fun)))(npr.randn(2)).shape == (2,2,2,2,2)

    check_grads(lambda x: np.sum(np.sin(jacobian(fun)(x))))(rng.randn(2))
    check_grads(lambda x: np.sum(np.sin(jacobian(jacobian(fun))(x))))(rng.randn(2))
