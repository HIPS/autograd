from autograd.tracer import primitive, getval
from autograd.extend import defvjp
from autograd.test_util import check_grads
from nose.tools import assert_raises_regexp

def test_check_vjp_1st_order_fail():
    @primitive
    def foo(x):
        return x * 2.0
    defvjp(foo, lambda ans, x : lambda g: g * 2.001)

    assert_raises_regexp(AssertionError,
                         "\(VJP\) check of foo failed",
                         lambda: check_grads(foo, modes=['rev'])(1.0))

def test_check_vjp_2nd_order_fail():
    @primitive
    def foo(x):
        return x * 2.0
    defvjp(foo, lambda ans, x : lambda g: bar(g) * 2)

    @primitive
    def bar(x):
        return x
    defvjp(bar, lambda ans, x : lambda g: g * 1.001)

    assert_raises_regexp(AssertionError,
                         "\(VJP\) check of vjp_foo failed",
                         lambda: check_grads(foo, modes=['rev'])(1.0))
