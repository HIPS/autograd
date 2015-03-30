import autograd.numpy.random as npr
import autograd.numpy as np
import operator as op
from numpy_utils import combo_check, stat_check, unary_ufunc_check, binary_ufunc_check

# Array statistics functions
def test_max(): stat_check(np.max)

# Unary ufunc tests

def test_sin(): unary_ufunc_check(np.sin) 

# Binary ufunc tests

def test_add(): binary_ufunc_check(np.add)

# Misc tests

def test_dot(): combo_check(np.dot, [0, 1],
                           [1.5, npr.randn(1, 3), npr.randn(2, 3), npr.randn(2, 4, 3)],
                           [0.3, npr.randn(3, 1), npr.randn(3, 4), npr.randn(2, 3, 4)])
