import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.util import flatten

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