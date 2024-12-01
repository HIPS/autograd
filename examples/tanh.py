import matplotlib.pyplot as plt

import autograd.numpy as np
from autograd import elementwise_grad as egrad

"""
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
"""


def tanh(x):
    return (1.0 - np.exp((-2 * x))) / (1.0 + np.exp(-(2 * x)))


### Plotting
plt.figure(figsize=(12, 8))
x = np.linspace(-7, 7, 700)
plt.plot(x, tanh(x), label="tanh(x)")
plt.plot(x, egrad(tanh)(x), label="1st derivative")
plt.plot(x, egrad(egrad(tanh))(x), label="2nd derivative")
plt.plot(x, egrad(egrad(egrad(tanh)))(x), label="3rd derivative")
plt.plot(x, egrad(egrad(egrad(egrad(tanh))))(x), label="4th derivative")
plt.xlabel("x")
plt.ylabel("y")
plt.ylim(-5, 5)
plt.yticks(np.arange(-5, 6, 1))
plt.legend()
plt.grid(True)
plt.title("tanh(x) and its derivatives")
plt.savefig("tanh.png")
plt.show()
