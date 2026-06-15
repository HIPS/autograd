import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad, make_vjp
from autograd.misc import const_graph, flatten
from autograd.test_util import scalar_close
from autograd.tracer import primitive


def test_const_graph():
    L = []

    def foo(x, y):
        L.append(None)
        return grad(lambda x: np.sin(x) + x * 2)(x * y)

    foo_wrapped = const_graph(foo)

    assert len(L) == 0
    assert scalar_close(foo(0.0, 0.0), foo_wrapped(0.0, 0.0))
    assert len(L) == 2
    assert scalar_close(foo(1.0, 0.5), foo_wrapped(1.0, 0.5))
    assert len(L) == 3
    assert scalar_close(foo(1.0, 0.5), foo_wrapped(1.0, 0.5))
    assert len(L) == 4


def test_const_graph_args():
    L = []

    @primitive
    def process(var, varname):
        L.append(varname)
        return var

    def foo(x, y, z):
        x = process(x, "x")
        y = process(y, "y")
        z = process(z, "z")
        return x + 2 * y + 3 * z

    foo_wrapped = const_graph(foo, 1.0, z=3.0)

    assert L == []
    assert scalar_close(foo(1.0, 2.0, 3.0), foo_wrapped(2.0))
    assert L == ["x", "y", "z", "x", "y", "z"]
    L = []
    assert scalar_close(foo(1.0, 2.0, 3.0), foo_wrapped(2.0))
    assert L == ["x", "y", "z", "y"]
    L = []
    assert scalar_close(foo(1.0, 2.0, 3.0), foo_wrapped(2.0))
    assert L == ["x", "y", "z", "y"]


def test_flatten():
    r = np.random.randn
    x = (1.0, r(2, 3), [r(1, 4), {"x": 2.0, "y": r(4, 2)}])
    x_flat, unflatten = flatten(x)
    assert x_flat.shape == (20,)
    assert x_flat[0] == 1.0
    assert np.all(x_flat == flatten(unflatten(x_flat))[0])

    y = (1.0, 2.0, [3.0, {"x": 2.0, "y": 4.0}])
    y_flat, unflatten = flatten(y)
    assert y_flat.shape == (5,)
    assert y == unflatten(y_flat)


def test_flatten_empty():
    val = (npr.randn(4), [npr.randn(3, 4), 2.5], (), (2.0, [1.0, npr.randn(2)]))
    vect, unflatten = flatten(val)
    val_recovered = unflatten(vect)
    vect_2, _ = flatten(val_recovered)
    assert np.all(vect == vect_2)


def test_flatten_dict():
    val = {"k": npr.random((4, 4)), "k2": npr.random((3, 3)), "k3": 3.0, "k4": [1.0, 4.0, 7.0, 9.0]}

    vect, unflatten = flatten(val)
    val_recovered = unflatten(vect)
    vect_2, _ = flatten(val_recovered)
    assert np.all(vect == vect_2)


def unflatten_tracing():
    val = [npr.randn(4), [npr.randn(3, 4), 2.5], (), (2.0, [1.0, npr.randn(2)])]
    vect, unflatten = flatten(val)

    def f(vect):
        return unflatten(vect)

    flatten2, _ = make_vjp(f)(vect)
    assert np.all(vect == flatten2(val))


def test_flatten_nodes_in_containers():
    # see issue #232
    def f(x, y):
        xy, _ = flatten([x, y])
        return np.sum(xy)

    grad(f)(1.0, 2.0)


def test_flatten_complex():
    val = 1 + 1j
    flat, unflatten = flatten(val)
    assert np.all(val == unflatten(flat))


### Some tests for retrieval of objects from object-like arrays


def test_object_array():
    x = object()
    a = np.array([x])
    assert a.item() is x


# Nested lists of objects and object arrays
def test_object_array_nested():
    x = object()
    y = object()
    a = np.array([[x], [y]])
    ab = np.array([[x, y], [y, x]])

    assert a[0, 0] is x
    assert a[1, 0] is y

    assert ab[0, 0] is x
    assert ab[0, 1] is y
    assert ab[1, 0] is y
    assert ab[1, 1] is x

    # Test mixed nesting; we use object arrays
    # for inhomogeneous shapes
    b = np.array([x, [y]], dtype=object)
    assert b[0] is x
    assert b[1][0] is y


def test_zero_dim_arrays():
    # 1. numeric scalar array
    x = np.array(5)
    arr = np.array([x])
    assert arr[0] == 5

    # 2. boolean scalar array
    y = np.array(True)
    arr2 = np.array([y])
    assert arr2[0] == True  # noqa: E712 because np.True_ is not a bool


def test_mixed_object_arrays():
    x = object()
    y = "string"
    z = 42
    arr = np.array([x, y, z])
    assert arr[0] is x
    assert arr[1] == "string"
    assert arr[2] == 42


def test_object_array_empty():
    a = np.array([])
    assert a.shape == (0,)
    b = np.array([[]])
    assert b.shape == (1, 0)
