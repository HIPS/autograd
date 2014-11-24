
import numpy as np
import matplotlib.pyplot as plt
from funkyyak import kyapply, grad # These are the only two functions exposed.
k = kyapply # kyapply(fun, *args) applies `fun` to `*args`. `k` is shorthand.

# Define a function capable of taking `Node` objects
def tanh(x):
    return (1.0 - k(np.exp, -x))  / ( 1.0 + k(np.exp, -x))

d_fun = grad(tanh)  # First derivative
dd_fun = grad(d_fun) # Second derivative
ddd_fun = grad(dd_fun) # Third derivative
dddd_fun = grad(ddd_fun) # Fourth derivative
ddddd_fun = grad(dddd_fun) # Fifth derivative
dddddd_fun = grad(ddddd_fun) # Sixth derivative

x = np.linspace(-7, 7, 200)
plt.plot(x, map(tanh, x),
         x, map(d_fun, x),
         x, map(dd_fun, x),
         x, map(ddd_fun, x),
         x, map(dddd_fun, x),
         x, map(ddddd_fun, x),
         x, map(dddddd_fun, x))

plt.axis('off')
plt.savefig("tanh.png")
plt.clf()
