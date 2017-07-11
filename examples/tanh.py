from __future__ import absolute_import
import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad

'''
Mathematically we can only take gradients of scalar-valued functions, but
autograd's grad function also handles numpy's familiar vectorization of scalar
functions, which is used in this example.

To be precise, grad(fun)(x) always returns the value of a vector-Jacobian
product, where the Jacobian of fun is evaluated at x and the the vector is an
all-ones vector with the same size as the output of fun. When vectorizing a
scalar-valued function over many arguments, the Jacobian of the overall
vector-to-vector mapping is diagonal, and so this vector-Jacobian product simply
returns the diagonal elements of the Jacobian, which is the gradient of the
function at each input value over which the function is vectorized.
'''

def tanh(x):
    return (1.0 - np.exp(-x))  / (1.0 + np.exp(-x))

x = np.linspace(-7, 7, 200)
plt.plot(x, tanh(x),
         x, grad(tanh)(x),                                 # first derivative
         x, grad(grad(tanh))(x),                           # second derivative
         x, grad(grad(grad(tanh)))(x),                     # third derivative
         x, grad(grad(grad(grad(tanh))))(x),               # fourth derivative
         x, grad(grad(grad(grad(grad(tanh)))))(x),         # fifth derivative
         x, grad(grad(grad(grad(grad(grad(tanh))))))(x))   # sixth derivative

plt.axis('off')
plt.savefig("tanh.png")
plt.show()
