from __future__ import absolute_import
import warnings

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.util import *
from autograd import grad

npr.seed(1)

def test_interp():
    x = np.arange(10) * 1.0
    xp = np.arange(10) * 1.0
    yp = np.arange(10) * 1.0
    def fun(yp): return to_scalar(np.interp(x, xp, yp))
    def dfun(yp): return to_scalar(grad(fun)(yp))
    print(fun(yp), dfun(yp))
    check_grads(fun, yp)
    check_grads(dfun, yp)
