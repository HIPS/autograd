"""This file doesn't import the numpy wrapper, to check if core works
   on basic operations even without numpy."""
from autograd import grad

# Non-numpy gradient checking functions.
def nd(f, x, eps=1e-4):
    return (f(x + eps/2) - f(x - eps/2)) / eps

def check_close(a, b, atol=1e-4, rtol=1e-4):
    assert abs(a - b) < atol + rtol*abs(b), "Diffs are: {0}".format(a - b)

def check_binary_func(fun):
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

def test_eq(): check_binary_func(lambda  x, y: x == y)
def test_neq(): check_binary_func(lambda x, y: x != y)
def test_leq(): check_binary_func(lambda x, y: x <= y)
def test_geq(): check_binary_func(lambda x, y: x >= y)
def test_lt(): check_binary_func(lambda  x, y: x < y)
def test_gt(): check_binary_func(lambda  x, y: x > y)


def test_return_both():
    fun = lambda x : 3.0 * x**3.2
    d_fun = grad(fun)
    f_and_d_fun = grad(fun, return_function_value=True)

    test_x = 1.7
    f, d = f_and_d_fun(test_x)
    assert f == fun(test_x)
    assert d == d_fun(test_x)
