"""This example shows how to define the gradient of your own functions.
This can be useful for speed, numerical stability, or in cases where
your code depends on external library calls."""
from __future__ import absolute_import
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr

from autograd import grad
from autograd.extend import primitive, defvjp
from autograd.test_util import check_grads


# @primitive tells Autograd not to look inside this function, but instead
# to treat it as a black box, whose gradient might be specified later.
# Functions with this decorator can contain anything that Python knows
# how to execute, and you can do things like in-place operations on arrays.
@primitive
def logsumexp(x):
    """Numerically stable log(sum(exp(x))), also defined in scipy.special"""
    max_x = np.max(x)
    return max_x + np.log(np.sum(np.exp(x - max_x)))

# Next, we write a function that specifies the gradient with a closure.
# The reason for the closure is so that the gradient can depend
# on both the input to the original function (x), and the output of the
# original function (ans).

def logsumexp_vjp(ans, x):
    # If you want to be able to take higher-order derivatives, then all the
    # code inside this function must be itself differentiable by Autograd.
    # This closure multiplies g with the Jacobian of logsumexp (d_ans/d_x).
    # Because Autograd uses reverse-mode differentiation, g contains
    # the gradient of the objective w.r.t. ans, the output of logsumexp.
    # This returned VJP function doesn't close over `x`, so Python can
    # garbage-collect `x` if there are no references to it elsewhere.
    x_shape = x.shape
    return lambda g: np.full(x_shape, g) * np.exp(x - np.full(x_shape, ans))

# Now we tell Autograd that logsumexmp has a gradient-making function.
defvjp(logsumexp, logsumexp_vjp)

if __name__ == '__main__':
    # Now we can use logsumexp() inside a larger function that we want
    # to differentiate.
    def example_func(y):
        z = y**2
        lse = logsumexp(z)
        return np.sum(lse)

    grad_of_example = grad(example_func)
    print("Gradient: \n", grad_of_example(npr.randn(10)))

    # Check the gradients numerically, just to be safe.
    check_grads(example_func, modes=['rev'])(npr.randn(10))
