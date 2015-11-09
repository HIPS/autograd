from __future__ import absolute_import
import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import elementwise_grad

# Here we use elementwise_grad to support broadcasting, which makes evaluating
# the gradient functions faster and avoids the need for calling 'map'.

def tanh(x):
    return (1.0 - np.exp(-x))  / (1.0 + np.exp(-x))

d_fun      = elementwise_grad(tanh)       # First derivative
dd_fun     = elementwise_grad(d_fun)      # Second derivative
ddd_fun    = elementwise_grad(dd_fun)     # Third derivative
dddd_fun   = elementwise_grad(ddd_fun)    # Fourth derivative
ddddd_fun  = elementwise_grad(dddd_fun)   # Fifth derivative
dddddd_fun = elementwise_grad(ddddd_fun)  # Sixth derivative

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
