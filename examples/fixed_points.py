from __future__ import print_function
import autograd.numpy as np
from autograd import grad
from autograd.misc.fixed_points import fixed_point

def newton_sqrt_iter(a):
    return lambda x: 0.5 * (x + a / x)

def grad_descent_sqrt_iter(a):
    return lambda x: x - 0.05 * (x**2 - a)

def sqrt(a, guess=10.):
    # return fixed_point(newton_sqrt_iter, a, guess, distance, 1e-4)
    return fixed_point(grad_descent_sqrt_iter, a, guess, distance, 1e-4)

def distance(x, y): return np.abs(x - y)

print(np.sqrt(2.))
print(sqrt(2.))
print()
print(grad(np.sqrt)(2.))
print(grad(sqrt)(2.))
print()
print(grad(grad(np.sqrt))(2.))
print(grad(grad(sqrt))(2.))
print()
