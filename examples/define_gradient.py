"""This example shows how to define the gradient of your own functions.
This can be useful for speed, numerical stability, or in cases where
your code depends on external library calls."""
from __future__ import absolute_import
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr

from autograd import grad, primitive
from autograd.util import quick_grad_check


# @primitive tells autograd not to look inside this function, but instead
# to treat it as a black box, whose gradient might be specified later.
# Functions with this decorator can contain anything that Python knows
# how to execute.
@primitive
def logsumexp(x):
    """Numerically stable log(sum(exp(x))), also defined in scipy.misc"""
    max_x = np.max(x)
    return max_x + np.log(np.sum(np.exp(x - max_x)))

# Next, we write a function that specifies the gradient with a closure.
# The reason for the closure is so that the gradient can depend
# on both the input to the original function (x), and the output of the
# original function (ans).

def logsumexp_vjp(g, ans, vs, gvs, x):
    # If you want to be able to take higher-order derivatives, then all the
    # code inside this function must be itself differentiable by autograd.
    # This closure multiplies g with the Jacobian of logsumexp (d_ans/d_x).
    # Because autograd uses reverse-mode differentiation, g contains
    # the gradient of the objective w.r.t. ans, the output of logsumexp.
    # The arguments `vs` and `gvs` contain information about the shapes of
    # `x` and `g` but using them is optional.
    return np.full(x.shape, g) * np.exp(x - np.full(x.shape, ans))

# Now we tell autograd that logsumexmp has a gradient-making function.
logsumexp.defvjp(logsumexp_vjp)

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
    quick_grad_check(example_func, npr.randn(10))
