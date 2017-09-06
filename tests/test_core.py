"""This file doesn't import the numpy wrapper, to check if core works
   on basic operations even without numpy."""
from __future__ import absolute_import
import warnings
from autograd.core import make_vjp
from autograd.wrap_util import unary_to_nary

@unary_to_nary
def grad(fun, x):
    vjp, _ = make_vjp(fun, x)
    return vjp(1.0)

# Non-numpy gradient checking functions.
def nd(f, x, eps=1e-4):
    return (f(x + eps/2) - f(x - eps/2)) / eps

def check_close(a, b, atol=1e-4, rtol=1e-4):
    assert abs(a - b) < atol + rtol*abs(b), "Diffs are: {0}".format(a - b)

def check_binary_func(fun, independent=False):
    with warnings.catch_warnings(record=independent) as w:
        x, y = 0.7, 1.8
        a = grad(fun)(x, y)
        b = nd(lambda x: fun(x, y), x)
        check_close(a, b)

        a = grad(fun, 1)(x, y)
        b = nd(lambda y: fun(x, y), y)
        check_close(a, b)

def test_add(): check_binary_func(lambda x, y: x + y)
def test_sub(): check_binary_func(lambda x, y: x - y)
def test_div(): check_binary_func(lambda x, y: x / y)
def test_mul(): check_binary_func(lambda x, y: x * y)
def test_pow(): check_binary_func(lambda x, y: x ** y)
def test_mod(): check_binary_func(lambda x, y: x % y)

def test_eq(): check_binary_func(lambda  x, y: x == y, independent=True)
def test_neq(): check_binary_func(lambda x, y: x != y, independent=True)
def test_leq(): check_binary_func(lambda x, y: x <= y, independent=True)
def test_geq(): check_binary_func(lambda x, y: x >= y, independent=True)
def test_lt(): check_binary_func(lambda  x, y: x < y, independent=True)
def test_gt(): check_binary_func(lambda  x, y: x > y, independent=True)
