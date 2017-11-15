from __future__ import absolute_import
import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import elementwise_grad as egrad

'''
Mathematically we can only take gradients of scalar-valued functions, but
autograd's elementwise_grad function also handles numpy's familiar vectorization
of scalar functions, which is used in this example.

To be precise, elementwise_grad(fun)(x) always returns the value of a
vector-Jacobian product, where the Jacobian of fun is evaluated at x and the
vector is an all-ones vector with the same size as the output of fun. When
vectorizing a scalar-valued function over many arguments, the Jacobian of the
overall vector-to-vector mapping is diagonal, and so this vector-Jacobian
product simply returns the diagonal elements of the Jacobian, which is the
(elementwise) gradient of the function at each input value over which the
function is vectorized.
'''

def tanh(x):
    return (1.0 - np.exp(-x))  / (1.0 + np.exp(-x))

x = np.linspace(-7, 7, 200)
plt.plot(x, tanh(x),
         x, egrad(tanh)(x),                                     # first derivative
         x, egrad(egrad(tanh))(x),                              # second derivative
         x, egrad(egrad(egrad(tanh)))(x),                       # third derivative
         x, egrad(egrad(egrad(egrad(tanh))))(x),                # fourth derivative
         x, egrad(egrad(egrad(egrad(egrad(tanh)))))(x),         # fifth derivative
         x, egrad(egrad(egrad(egrad(egrad(egrad(tanh))))))(x))  # sixth derivative

plt.axis('off')
plt.savefig("tanh.png")
plt.show()
