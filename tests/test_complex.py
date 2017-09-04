from __future__ import absolute_import
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.test_util import check_grads
from autograd import grad
npr.seed(1)

def test_real_type():
    fun = lambda x: np.sum(np.real(x))
    df = grad(fun)
    assert np.isrealobj(df(2.0))
    assert np.iscomplexobj(df(1.0j))

def test_real_if_close_type():
    fun = lambda x: np.sum(np.real(x))
    df = grad(fun)
    assert np.isrealobj(df(1.0))
    assert np.iscomplexobj(df(1.0j))

def test_angle_real():
    fun = lambda x : np.angle(x)
    d_fun = lambda x: grad(fun)(x)
    check_grads(fun)(npr.rand())
    check_grads(d_fun)(npr.rand())

def test_angle_complex():
    fun = lambda x : np.angle(x)
    d_fun = lambda x: grad(fun)(x)
    check_grads(fun)(npr.rand() + 1j*npr.rand())
    check_grads(d_fun)(npr.rand() + 1j*npr.rand())

def test_abs_real():
    fun = lambda x : np.abs(x)
    d_fun = lambda x: grad(fun)(x)
    check_grads(fun)(1.1)
    check_grads(d_fun)(2.1)

def test_abs_complex():
    fun = lambda x : np.abs(x)
    d_fun = lambda x: grad(fun)(x)
    check_grads(fun)(1.1 + 1.2j)
    check_grads(d_fun)(1.1 + 1.3j)
