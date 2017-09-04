# This file is to check that future division works.
from __future__ import division
from __future__ import absolute_import

import autograd.numpy as np
from autograd.test_util import check_grads
from autograd import grad
from test_binary_ops import arg_pairs

def test_div():
    fun = lambda x, y : x / y
    make_gap_from_zero = lambda x : np.sqrt(x **2 + 0.5)
    for arg1, arg2 in arg_pairs():
        arg1 = make_gap_from_zero(arg1)
        arg2 = make_gap_from_zero(arg2)
        check_grads(fun)(arg1, arg2)
