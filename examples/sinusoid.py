from __future__ import absolute_import
from __future__ import print_function
import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad
from builtins import range, map

def fun(x):
    return np.sin(x)

d_fun = grad(fun)    # First derivative
dd_fun = grad(d_fun) # Second derivative

x = np.linspace(-10, 10, 100)
plt.plot(x, list(map(fun, x)), x, list(map(d_fun, x)), x, list(map(dd_fun, x)))

plt.xlim([-10, 10])
plt.ylim([-1.2, 1.2])
plt.axis('off')
plt.savefig("sinusoid.png")
plt.clf()

# Taylor approximation to sin function
def fun(x):
    currterm = x
    ans = currterm
    for i in range(1000):
        print(i, end=' ')
        currterm = - currterm * x ** 2 / ((2 * i + 3) * (2 * i + 2))
        ans = ans + currterm
        if np.abs(currterm) < 0.2: break # (Very generous tolerance!)

    return ans

d_fun = grad(fun)
dd_fun = grad(d_fun)

x = np.linspace(-10, 10, 100)
plt.plot(x, list(map(fun, x)), x, list(map(d_fun, x)), x, list(map(dd_fun, x)))

plt.xlim([-10, 10])
plt.ylim([-1.2, 1.2])
plt.axis('off')
plt.savefig("sinusoid_taylor.png")
plt.clf()
