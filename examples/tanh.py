from __future__ import absolute_import
import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad

'''
Mathematically we can only take gradients of scalar-valued functions, but
autograd's grad function also handles numpy's familiar broadcasting behavior,
which is used in this example.

To be precise, grad(fun)(x) always returns the value of a vector-Jacobian
product, where the Jacobian of fun is evaluated at x and the the vector is an
all-ones vector with the same size as the output of fun. When broadcasting a
scalar-valued function over many arguments, the Jacobian of the overall
vector-to-vector mapping is diagonal, and so this vector-Jacobian product simply
returns the diagonal elements of the Jacobian, which is the gradient of the
function at each broadcasted input value.
'''

def tanh(x):
    return (1.0 - np.exp(-x))  / (1.0 + np.exp(-x))

d_fun      = grad(tanh)       # First derivative
dd_fun     = grad(d_fun)      # Second derivative
ddd_fun    = grad(dd_fun)     # Third derivative
dddd_fun   = grad(ddd_fun)    # Fourth derivative
ddddd_fun  = grad(dddd_fun)   # Fifth derivative
dddddd_fun = grad(ddddd_fun)  # Sixth derivative

x = np.linspace(-7, 7, 200)
plt.plot(x, tanh(x),
         x, d_fun(x),
         x, dd_fun(x),
         x, ddd_fun(x),
         x, dddd_fun(x),
         x, ddddd_fun(x),
         x, dddddd_fun(x))

plt.axis('off')
plt.savefig("tanh.png")
plt.show()
