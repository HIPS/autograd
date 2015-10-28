from __future__ import absolute_import
from __future__ import print_function
from autograd import grad
import autograd.numpy as np
import autograd.numpy.random as npr
from time import time
from contextlib import contextmanager
from builtins import range

@contextmanager
def tictoc(text=""):
    print("--- Start clock ---")
    t1 = time()
    yield
    dt = time() - t1
    print("--- Stop clock {0}: {1} seconds elapsed ---".format(text, dt))

def fan_out_fan_in():
    """The 'Pearlmutter test' """
    def fun(x):
        for i in range(10**4):
            x = (x + x)/2.0
        return np.sum(x)

    with tictoc():
        grad(fun)(1.0)

def convolution():
    # MNIST-scale convolution operation
    import autograd.scipy.signal
    convolve = autograd.scipy.signal.convolve
    dat = npr.randn(256, 3, 28, 28)
    kernel = npr.randn(3, 5, 5)
    with tictoc():
        convolve(dat, kernel, axes=([2, 3], [1, 2]), dot_axes=([1], [0]))

def dot_equivalent():
    # MNIST-scale convolution operation
    import autograd.scipy.signal
    dat = npr.randn(256, 3, 24, 5, 24, 5)
    kernel = npr.randn(3, 5, 5)
    with tictoc():
        np.tensordot(dat, kernel, axes=[(1, 3, 5), (0, 1, 2)])


# fan_out_fan_in()
# convolution()
dot_equivalent()
