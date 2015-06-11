from __future__ import absolute_import
from __future__ import print_function
import autograd.numpy.random as npr
import autograd.numpy as np
import operator as op
from autograd.numpy.use_gpu_numpy import use_gpu_numpy
from numpy_utils import (combo_check, stat_check, unary_ufunc_check,
                         binary_ufunc_check, binary_ufunc_check_no_same_args)

npr.seed(0)

if not use_gpu_numpy():
    print("Can't test GPU support without flag set")
else:

    def R(*shape):
        arr = npr.randn(*shape)
        return np.array(arr, dtype=np.gpu_float32)

    def test_dot(): combo_check(np.dot, [0, 1],
                                [R(2, 3)],
                                [R(3, 4)])
                                # [1.5, R(3), R(2, 3)],
                                # [0.3, R(3), R(3, 4)])
