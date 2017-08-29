import autograd.numpy as np
from autograd.test_util import scalar_close
from autograd import grad
from autograd.tracer import primitive
from autograd.misc import const_graph, flatten

def test_const_graph():
    L = []
    def foo(x, y):
        L.append(None)
        return grad(lambda x: np.sin(x) + x * 2)(x * y)

    foo_wrapped = const_graph(foo)

    assert len(L) == 0
    assert scalar_close(foo(0., 0.),
                        foo_wrapped(0., 0.))
    assert len(L) == 2
    assert scalar_close(foo(1., 0.5),
                        foo_wrapped(1., 0.5))
    assert len(L) == 3
    assert scalar_close(foo(1., 0.5),
                        foo_wrapped(1., 0.5))
    assert len(L) == 4

def test_const_graph_args():
    L = []

    @primitive
    def process(var, varname):
        L.append(varname)
        return var

    def foo(x, y, z):
        x = process(x, 'x')
        y = process(y, 'y')
        z = process(z, 'z')
        return x + 2*y + 3*z

    foo_wrapped = const_graph(foo, 1., z=3.)

    assert L == []
    assert scalar_close(foo(1., 2., 3.),
                        foo_wrapped(2.))
    assert L == ['x', 'y', 'z', 'x', 'y', 'z']
    L = []
    assert scalar_close(foo(1., 2., 3.),
                        foo_wrapped(2.))
    assert L == ['x', 'y', 'z', 'y']
    L = []
    assert scalar_close(foo(1., 2., 3.),
                        foo_wrapped(2.))
    assert L == ['x', 'y', 'z', 'y']

def test_flatten():
    r = np.random.randn
    x = (1.0, r(2,3), [r(1,4), {'x': 2.0, 'y': r(4,2)}])
    x_flat, unflatten = flatten(x)
    assert x_flat.shape == (20,)
    assert x_flat[0] == 1.0
    assert np.all(x_flat == flatten(unflatten(x_flat))[0])
