"""
Basic tests for forward mode autodiff.
"""
from autograd.util import check_forward_grads, check_grads
from autograd import grad, forward_mode_grad
import numpy as np


def test_fwd():
    def f(x, y):
        return x ** 3 + x * y + y
    check_forward_grads(f, 1.23, 4.56)

def test_fwd_fwd():
    def f(x, y):
        return x ** 3 + x * y + y ** 4

    f_1 = forward_mode_grad(f)
    f_2 = forward_mode_grad(f, argnum=1)

    check_forward_grads(f_1, 1.23, 4.56)
    check_forward_grads(f_2, 1.23, 4.56)

def test_rev_fwd():
    def f(x, y):
        return x ** 3 + x * y + y ** 4

    f_1 = forward_mode_grad(f)
    f_2 = forward_mode_grad(f, argnum=1)

    check_grads(f_1, 1.23, 4.56)
    check_grads(f_2, 1.23, 4.56)

def test_fwd_rev():
    def f(x, y):
        return x ** 3 + x * y + y ** 4

    f_1 = grad(f)
    f_2 = grad(f, argnum=1)

    check_forward_grads(f_1, 1.23, 4.56)
    check_forward_grads(f_2, 1.23, 4.56)
