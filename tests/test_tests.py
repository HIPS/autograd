from autograd.tracer import primitive, getval
from autograd.extend import defvjp
from autograd.test_util import check_grads
from pytest import raises

def test_check_vjp_1st_order_fail():
    @primitive
    def foo(x):
        return x * 2.0
    defvjp(foo, lambda ans, x : lambda g: g * 2.001)

    with raises(AssertionError, match="\\(VJP\\) check of foo failed"):
         check_grads(foo, modes=['rev'])(1.0)

def test_check_vjp_2nd_order_fail():
    @primitive
    def foo(x):
        return x * 2.0
    defvjp(foo, lambda ans, x : lambda g: bar(g) * 2)

    @primitive
    def bar(x):
        return x
    defvjp(bar, lambda ans, x : lambda g: g * 1.001)

    with raises(AssertionError, match="\\(VJP\\) check of vjp_foo failed"):
         check_grads(foo, modes=['rev'])(1.0)
