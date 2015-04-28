import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.util import *
from autograd import (grad, elementwise_grad, jacobian, value_and_grad,
                      grad_and_aux, hessian_vector_product, hessian)
npr.seed(1)

def test_hessian():
    # Check Hessian of a quadratic function.
    D = 5
    H = npr.randn(D, D)
    def fun(x):
        return np.dot(np.dot(x, H),x)
    hess = hessian(fun)
    x = npr.randn(D)
    check_equivalent(hess(x), H + H.T)
