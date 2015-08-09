from __future__ import absolute_import
from __future__ import print_function
import autograd.numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize

def rosenbrock(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

# Build a function that also returns gradients using autograd.
rosenbrock_with_grad = value_and_grad(rosenbrock)

# Optimize using conjugate gradients.
result = minimize(rosenbrock_with_grad, x0=np.array([0.0, 0.0]), jac=True, method='CG')
print("Found minimum at {0}".format(result.x))
