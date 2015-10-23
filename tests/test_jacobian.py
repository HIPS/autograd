from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.util import *
from autograd import grad, jacobian
from autograd.convenience_wrappers import jacobian as old_jacobian
npr.seed(1)


def test_jacobian_against_grad():
    fun = lambda x: np.sum(np.sin(x), axis=1, keepdims=True)
    A = npr.randn(1,3)
    assert np.allclose(grad(fun)(A), jacobian(fun)(A))

def test_jacobian_scalar_to_vector():
    fun = lambda x: np.array([x, x**2, x**3])
    val = npr.randn()
    assert np.allclose(jacobian(fun)(val), np.array([1., 2*val, 3*val**2]))

def test_jacobian_against_stacked_grads():
    scalar_funs = [
        lambda x: np.sum(x**3),
        lambda x: np.prod(np.sin(x) + np.sin(x)),
        lambda x: grad(lambda y: np.exp(y) * np.tanh(x[0]))(x[1])
    ]

    vector_fun = lambda x: np.array([f(x) for f in scalar_funs])

    x = npr.randn(5)
    jac = jacobian(vector_fun)(x)
    grads = [grad(f)(x) for f in scalar_funs]

    assert np.allclose(jac, np.vstack(grads))

def test_jacobian_against_wrapper():
    A = npr.randn(3,3,3)
    fun = lambda x: np.einsum(
        'ijk,jkl->il',
        A, np.sin(x[...,None] * np.tanh(x[None,...])))

    B = npr.randn(3,3)
    jac1 = jacobian(fun)(B)
    jac2 = old_jacobian(fun)(B)

    assert np.allclose(jac1, jac2)

def test_jacobian_higher_order():
    fun = lambda x: np.sin(np.outer(x,x)) + np.cos(np.dot(x,x))

    jacobian(fun)(npr.randn(3)).shape == (3,3,3)
    jacobian(jacobian(fun))(npr.randn(3)).shape == (3,3,3,3)
    jacobian(jacobian(jacobian(fun)))(npr.randn(3)).shape == (3,3,3,3,3)

    check_grads(lambda x: np.sum(jacobian(fun)(x)), npr.randn(3))
    check_grads(lambda x: np.sum(jacobian(jacobian(fun))(x)), npr.randn(3))
