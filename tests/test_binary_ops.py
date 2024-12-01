import itertools as it
import warnings

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad, value_and_grad
from autograd.test_util import check_grads

rs = npr.RandomState(0)


def arg_pairs():
    scalar = 2.0
    vector = rs.randn(4)
    mat = rs.randn(3, 4)
    mat2 = rs.randn(1, 4)
    allargs = [scalar, vector, mat, mat2]
    yield from it.product(allargs, allargs)


def test_mul():
    fun = lambda x, y: x * y
    for arg1, arg2 in arg_pairs():
        check_grads(fun)(arg1, arg2)


def test_add():
    fun = lambda x, y: x + y
    for arg1, arg2 in arg_pairs():
        check_grads(fun)(arg1, arg2)


def test_sub():
    fun = lambda x, y: x - y
    for arg1, arg2 in arg_pairs():
        check_grads(fun)(arg1, arg2)


def test_div():
    fun = lambda x, y: x / y
    make_gap_from_zero = lambda x: np.sqrt(x**2 + 0.5)
    for arg1, arg2 in arg_pairs():
        arg1 = make_gap_from_zero(arg1)
        arg2 = make_gap_from_zero(arg2)
        check_grads(fun)(arg1, arg2)


def test_mod():
    fun = lambda x, y: x % y
    make_gap_from_zero = lambda x: np.sqrt(x**2 + 0.5)
    for arg1, arg2 in arg_pairs():
        if arg1 is not arg2:  # Gradient undefined at x == y
            arg1 = make_gap_from_zero(arg1)
            arg2 = make_gap_from_zero(arg2)
            check_grads(fun)(arg1, arg2)


def test_pow():
    fun = lambda x, y: x**y
    make_positive = lambda x: np.abs(x) + 1.1  # Numeric derivatives fail near zero
    for arg1, arg2 in arg_pairs():
        arg1 = make_positive(arg1)
        check_grads(fun)(arg1, arg2)


def test_arctan2():
    for arg1, arg2 in arg_pairs():
        check_grads(np.arctan2)(arg1, arg2)


def test_hypot():
    for arg1, arg2 in arg_pairs():
        check_grads(np.hypot, modes=["rev"])(arg1, arg2)


def test_comparison_grads():
    compare_funs = [
        lambda x, y: np.sum(x < x) + 0.0,
        lambda x, y: np.sum(x <= y) + 0.0,
        lambda x, y: np.sum(x > y) + 0.0,
        lambda x, y: np.sum(x >= y) + 0.0,
        lambda x, y: np.sum(x == y) + 0.0,
        lambda x, y: np.sum(x != y) + 0.0,
    ]

    with warnings.catch_warnings(record=True) as w:
        for arg1, arg2 in arg_pairs():
            zeros = (arg1 + arg2) * 0  # get correct shape
            for fun in compare_funs:
                assert np.all(grad(fun)(arg1, arg2) == zeros)
                assert np.all(grad(fun, argnum=1)(arg1, arg2) == zeros)


def test_comparison_values():
    compare_funs = [
        lambda x, y: np.sum(x < x) + 0.0,
        lambda x, y: np.sum(x <= y) + 0.0,
        lambda x, y: np.sum(x > y) + 0.0,
        lambda x, y: np.sum(x >= y) + 0.0,
        lambda x, y: np.sum(x == y) + 0.0,
        lambda x, y: np.sum(x != y) + 0.0,
    ]

    for arg1, arg2 in arg_pairs():
        for fun in compare_funs:
            fun_val = fun(arg1, arg2)
            fun_val_from_grad, _ = value_and_grad(fun)(arg1, arg2)
            assert fun_val == fun_val_from_grad, (fun_val, fun_val_from_grad)
