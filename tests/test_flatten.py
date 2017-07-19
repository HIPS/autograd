import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.util import flatten
from autograd import make_vjp, grad

def test_flatten():
    val = (npr.randn(4), [npr.randn(3,4), 2.5], (), (2.0, [1.0, npr.randn(2)]))
    vect, unflatten = flatten(val)
    val_recovered = unflatten(vect)
    vect_2, _ = flatten(val_recovered)
    assert np.all(vect == vect_2)

def test_flatten_dict():
    val = {'k':  npr.random((4, 4)),
           'k2': npr.random((3, 3)),
           'k3': 3.0,
           'k4': [1.0, 4.0, 7.0, 9.0]}

    vect, unflatten = flatten(val)
    val_recovered = unflatten(vect)
    vect_2, _ = flatten(val_recovered)
    assert np.all(vect == vect_2)

def unflatten_tracing():
    val = [npr.randn(4), [npr.randn(3,4), 2.5], (), (2.0, [1.0, npr.randn(2)])]
    vect, unflatten = flatten(val)
    def f(vect): return unflatten(vect)
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
