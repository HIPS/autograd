from __future__ import absolute_import
import warnings

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.test_util import check_grads
from autograd import grad
from numpy_utils import combo_check

from dask.array.utils import assert_eq
import dask.array as da

npr.seed(1)

def test_dask():
    x = np.arange(10)
    xx = da.arange(10, chunks=(5,))

    assert_eq(x, xx)

    def f(x):
        return np.sin(x).sum()

    f_prime = grad(f)

    assert isinstance(f_prime(xx), type(xx))

    assert_eq(f_prime(x), f_prime(xx))
