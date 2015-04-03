import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.signal

from numpy_utils import combo_check
npr.seed(1)

R = npr.randn
def test_convolve(): combo_check(autograd.scipy.signal.convolve, [0,1],
                                    [R(4), R(5), R(6)],
                                    [R(2), R(3), R(4)],
                                    mode=['full', 'valid', 'same'])

def test_convolve2d(): combo_check(autograd.scipy.signal.convolve2d, [0, 1],
                                   [R(4, 3), R(5, 4), R(6, 7)],
                                   [R(2, 2), R(3, 2), R(4, 3)],
                                   mode=['full', 'valid', 'same'])

