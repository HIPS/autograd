"""Some standard gradient-based stochastic optimizers.

   These are just standard routines that don't make any use of autograd,
   though you could take gradients of these functions too if you want
   to do meta-optimization."""

import autograd.numpy as np


def sgd(grad, x, callback=None, num_iters=200, step_size=0.1, mass=0.9):
    """Stochastic gradient descent with momentum.
    grad() must have signature grad(x, i), where i is the iteration number."""
    velocity = np.zeros(len(x))
    for i in xrange(num_iters):
        g = grad(x, i)
        if callback: callback(x, i, g)
        velocity = mass * velocity - (1.0 - mass) * g
        x += step_size * velocity
    return x

def rmsprop(grad, x, callback=None, num_iters=100, step_size=0.1, gamma=0.9, eps = 10**-8):
    """Root mean squared prop: See Adagrad paper for details."""
    avg_sq_grad = np.ones(len(x))
    for i in xrange(num_iters):
        g = grad(x, i)
        if callback: callback(x, i, g)
        avg_sq_grad = avg_sq_grad * gamma + g**2 * (1 - gamma)
        x -= step_size * g/(np.sqrt(avg_sq_grad) + eps)
    return x

def adam(grad, x, callback=None, num_iters=100,
         step_size=0.1, b1 = 0.1, b2 = 0.01, eps = 10**-4, lam=10**-4):
    """Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
    It's basically RMSprop with momentum and some correction terms."""
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    for i in xrange(num_iters):
        b1t = 1 - (1 - b1) * lam**i
        g = grad(x, i)
        if callback: callback(x, i, g)
        m = b1t * g     + (1 - b1t) * m     # First  moment estimate.
        v = b2 * (g**2) + (1 - b2) * v      # Second moment estimate.
        mhat = m / (1 - (1 - b1)**(i + 1))  # Bias correction.
        vhat = v / (1 - (1 - b2)**(i + 1))
        x -= step_size*mhat/(np.sqrt(vhat) + eps)
    return x
