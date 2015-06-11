# This file is to check that future division works.
from __future__ import division
from __future__ import absolute_import

import autograd.numpy as np
from autograd.util import *
from autograd import grad
from test_binary_ops import arg_pairs

def test_div():
    fun = lambda x, y : to_scalar(x / y)
    d_fun_0 = lambda x, y : to_scalar(grad(fun, 0)(x, y))
    d_fun_1 = lambda x, y : to_scalar(grad(fun, 1)(x, y))
    make_gap_from_zero = lambda x : np.sqrt(x **2 + 0.5)
    for arg1, arg2 in arg_pairs():
        arg1 = make_gap_from_zero(arg1)
        arg2 = make_gap_from_zero(arg2)
        check_grads(fun, arg1, arg2)
        check_grads(d_fun_0, arg1, arg2)
        check_grads(d_fun_1, arg1, arg2)

