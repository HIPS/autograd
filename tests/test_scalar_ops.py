import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.test_util import check_grads


def test_abs():
    fun = lambda x: 3.0 * np.abs(x)
    check_grads(fun)(1.1)
    check_grads(fun)(-1.1)
    check_grads(fun, order=1)(0.0)


def test_absolute():
    fun = lambda x: 3.0 * np.absolute(x)
    check_grads(fun)(1.1)
    check_grads(fun)(-1.1)
    check_grads(fun, order=1)(0.0)


def test_sin():
    rng = npr.RandomState(42)
    fun = lambda x: 3.0 * np.sin(x)
    check_grads(fun)(rng.randn())


def test_sign():
    fun = lambda x: 3.0 * np.sign(x)
    check_grads(fun)(1.1)
    check_grads(fun)(-1.1)


def test_exp():
    rng = npr.RandomState(42)
    fun = lambda x: 3.0 * np.exp(x)
    check_grads(fun)(rng.randn())


def test_log():
    rng = npr.RandomState(42)
    fun = lambda x: 3.0 * np.log(x)
    check_grads(fun)(abs(rng.randn()))


def test_log2():
    rng = npr.RandomState(42)
    fun = lambda x: 3.0 * np.log2(x)
    check_grads(fun)(abs(rng.randn()))


def test_log10():
    rng = npr.RandomState(42)
    fun = lambda x: 3.0 * np.log10(x)
    check_grads(fun)(abs(rng.randn()))


def test_log1p():
    rng = npr.RandomState(42)
    fun = lambda x: 3.0 * np.log1p(x)
    check_grads(fun)(abs(rng.randn()))


def test_expm1():
    rng = npr.RandomState(42)
    fun = lambda x: 3.0 * np.expm1(x)
    check_grads(fun)(abs(rng.randn()))


def test_exp2():
    rng = npr.RandomState(42)
    fun = lambda x: 3.0 * np.exp2(x)
    check_grads(fun)(abs(rng.randn()))


def test_neg():
    rng = npr.RandomState(42)
    fun = lambda x: 3.0 * -x
    check_grads(fun)(rng.randn())


def test_cos():
    rng = npr.RandomState(42)
    fun = lambda x: 3.0 * np.cos(x)
    check_grads(fun)(rng.randn())


def test_tan():
    rng = npr.RandomState(42)
    fun = lambda x: 3.0 * np.tan(x)
    check_grads(fun)(rng.randn())


def test_cosh():
    rng = npr.RandomState(42)
    fun = lambda x: 3.0 * np.cosh(x)
    check_grads(fun)(rng.randn())


def test_sinh():
    rng = npr.RandomState(42)
    fun = lambda x: 3.0 * np.sinh(x)
    check_grads(fun)(rng.randn())


def test_tanh():
    rng = npr.RandomState(42)
    fun = lambda x: 3.0 * np.tanh(x)
    check_grads(fun)(rng.randn())


def test_arccos():
    fun = lambda x: 3.0 * np.arccos(x)
    check_grads(fun)(0.1)


def test_arcsin():
    fun = lambda x: 3.0 * np.arcsin(x)
    check_grads(fun)(0.1)


def test_arctan():
    fun = lambda x: 3.0 * np.arctan(x)
    check_grads(fun)(0.2)


def test_arccosh():
    rng = npr.RandomState(42)
    fun = lambda x: 3.0 * np.arccosh(x)
    check_grads(fun)(rng.randn() ** 2 + 1.2)


def test_arcsinh():
    rng = npr.RandomState(42)
    fun = lambda x: 3.0 * np.arcsinh(x)
    check_grads(fun)(rng.randn())


def test_arctanh():
    fun = lambda x: 3.0 * np.arctanh(x)
    check_grads(fun)(0.2)


def test_sqrt():
    rng = npr.RandomState(42)
    fun = lambda x: 3.0 * np.sqrt(x)
    check_grads(fun)(10.0 * rng.rand())


def test_power_arg0():
    rng = npr.RandomState(42)
    # the +1.'s here are to avoid regimes where numerical diffs fail
    make_fun = lambda y: lambda x: np.power(x, y)
    fun = make_fun(rng.randn() ** 2 + 1.0)
    check_grads(fun)(rng.rand() ** 2 + 1.0)

    # test y == 0. as a special case, c.f. #116
    fun = make_fun(0.0)
    assert grad(fun)(0.0) == 0.0
    assert grad(grad(fun))(0.0) == 0.0


def test_power_arg1():
    rng = npr.RandomState(42)
    x = rng.randn() ** 2
    fun = lambda y: np.power(x, y)
    check_grads(fun)(rng.rand() ** 2)


def test_power_arg1_zero():
    rng = npr.RandomState(42)
    fun = lambda y: np.power(0.0, y)
    check_grads(fun)(rng.rand() ** 2)


def test_mod_arg0():
    rng = npr.RandomState(42)
    fun = lambda x, y: np.mod(x, y)
    check_grads(fun)(rng.rand(), rng.rand())


def test_mod_arg1():
    rng = npr.RandomState(42)
    fun = lambda x, y: np.mod(x, y)
    check_grads(fun)(rng.rand(), rng.rand())


def test_divide_arg0():
    rng = npr.RandomState(42)
    fun = lambda x, y: np.divide(x, y)
    check_grads(fun)(rng.rand(), rng.rand())


def test_divide_arg1():
    rng = npr.RandomState(42)
    fun = lambda x, y: np.divide(x, y)
    check_grads(fun)(rng.rand(), rng.rand())


def test_multiply_arg0():
    rng = npr.RandomState(42)
    fun = lambda x, y: np.multiply(x, y)
    check_grads(fun)(rng.rand(), rng.rand())


def test_multiply_arg1():
    rng = npr.RandomState(42)
    fun = lambda x, y: np.multiply(x, y)
    check_grads(fun)(rng.rand(), rng.rand())


def test_true_divide_arg0():
    rng = npr.RandomState(42)
    fun = lambda x, y: np.true_divide(x, y)
    check_grads(fun)(rng.rand(), rng.rand())


def test_true_divide_arg1():
    rng = npr.RandomState(42)
    fun = lambda x, y: np.true_divide(x, y)
    check_grads(fun)(rng.rand(), rng.rand())


def test_reciprocal():
    rng = npr.RandomState(42)
    fun = lambda x: np.reciprocal(x)
    check_grads(fun)(rng.rand())


def test_negative():
    rng = npr.RandomState(42)
    fun = lambda x: np.negative(x)
    check_grads(fun)(rng.rand())


def test_rad2deg():
    rng = npr.RandomState(42)
    fun = lambda x: 3.0 * np.rad2deg(x)
    check_grads(fun)(10.0 * rng.rand())


def test_deg2rad():
    rng = npr.RandomState(42)
    fun = lambda x: 3.0 * np.deg2rad(x)
    check_grads(fun)(10.0 * rng.rand())


def test_radians():
    rng = npr.RandomState(42)
    fun = lambda x: 3.0 * np.radians(x)
    check_grads(fun)(10.0 * rng.rand())


def test_degrees():
    rng = npr.RandomState(42)
    fun = lambda x: 3.0 * np.degrees(x)
    check_grads(fun)(10.0 * rng.rand())


def test_sinc():
    rng = npr.RandomState(42)
    fun = lambda x: 3.0 * np.sinc(x)
    check_grads(fun)(10.0 * rng.rand())
