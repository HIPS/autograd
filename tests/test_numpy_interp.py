from __future__ import absolute_import
import warnings

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.util import *
from autograd import grad

from numpy import interp as ninterp
from numpy.testing import assert_allclose

def test_interp():
    x = np.arange(-20, 20, 0.1)
    xp = np.arange(10) * 1.0
    npr.seed(1)
    yp = xp ** 0.5 + npr.normal(size=xp.shape)
    def fun(yp): return to_scalar(np.interp(x, xp, yp))
    def dfun(yp): return to_scalar(grad(fun)(yp))

    check_grads(fun, yp)
    check_grads(dfun, yp)

def test_interp_edge():
    x = np.arange(-20, 20, 0.1)
    xp = np.arange(10) * 1.0
    npr.seed(1)
    yp = xp ** 0.5 + npr.normal(size=xp.shape)
    def fun(yp): return to_scalar(np.interp(x, xp, yp, left=-1, right=-1))
    def dfun(yp): return to_scalar(grad(fun)(yp))

    check_grads(fun, yp)
    check_grads(dfun, yp)

def test_interp_correctness():
    x = np.arange(-100, 100, 0.1)
    xp = np.arange(10) * 1.0
    npr.seed(1)
    yp = xp ** 0.5 + npr.normal(size=xp.shape)

    y1 = ninterp(x, xp, yp)
    y2 = np.interp(x, xp, yp)

    assert_allclose(y1, y2)

def test_interp_correctness_edge():
    x = np.arange(-100, 100, 0.1)
    xp = np.arange(10) * 1.0
    npr.seed(1)
    yp = xp ** 0.5 + npr.normal(size=xp.shape)

    y1 = ninterp(x, xp, yp, left=-1, right=-1)
    y2 = np.interp(x, xp, yp, left=-1, right=-1)

    assert_allclose(y1, y2)
