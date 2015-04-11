"""This example shows how to define the gradient of your own functions.
This can be useful for speed, numerical stability, or in cases where
your code depends on external library calls."""
import autograd.numpy as np
import autograd.numpy.random as npr

from autograd.core import grad, primitive
from autograd.util import quick_grad_check


# The @primitive decorator tells autograd not to look inside this function,
# but instead to treat it as a black box, whose gradient might be specified
# later.  Functions that have this decorator can contain anything that Python
# knows how to execute.
@primitive
def logsumexp(x):                     # also defined in scipy.misc
    return np.log(np.sum(np.exp(x)))

# This function will get called on the forward pass,
# and tells autograd how to compute the gradient on the reverse pass.
# For the purposes of computing second derivatives, all the code inside
# the gradient-making function must consist of functions known to autograd.
def make_grad_logsumexp(ans, x):
    # ans is the result of the forward computation, which we can re-use
    # in our gradient computation.
    def gradient_product(g):
        # Because autograd uses reverse-mode differentiation, g always contains
        #  the gradient of the objective w.r.t. the output of logsumexp.
        # This function multiplies g with the Jacobian of logsumexp.
        return np.full(x.shape, g) * np.exp(x - np.full(x.shape, ans))
    return gradient_product

# Now, the only thing left to do is to tell autograd that
# logsumexmp has a gradient-making function:
logsumexp.defgrad(make_grad_logsumexp)


if __name__ == '__main__':
    # Now we can use logsumexp() inside a larger function that we want
    # to differentiate.

    def example_func(y):
        z = y**2
        lse = logsumexp(z)
        return np.sum(lse)

    grad_of_example = grad(example_func)
    print "Gradient: ", grad_of_example(npr.randn(10))

    # Check the gradients numerically, just to be safe.
    quick_grad_check(example_func,npr.randn(10))
