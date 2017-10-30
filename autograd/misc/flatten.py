"""
Handy functions for flattening nested containers containing numpy
arrays. The main purpose is to make examples and optimizers simpler.
"""
from autograd import make_vjp
from autograd.builtins import type
import autograd.numpy as np

def flatten(value):
    """Flattens any nesting of tuples, lists, or dicts, with numpy arrays or
    scalars inside. Returns 1D numpy array and an unflatten function.
    Doesn't preserve mixed numeric types (e.g. floats and ints). Assumes dict
    keys are sortable."""
    unflatten, flat_value = make_vjp(_flatten)(value)
    return flat_value, unflatten

def _flatten(value):
    t = type(value)
    if t in (list, tuple):
        return _concatenate(map(_flatten, value))
    elif t is dict:
        return _concatenate(_flatten(value[k]) for k in sorted(value))
    else:
        return np.ravel(value)

def _concatenate(lst):
    lst = list(lst)
    return np.concatenate(lst) if lst else np.array([])

def flatten_func(func, example):
    _ex, unflatten = flatten(example)
    _func = lambda _x, *args: flatten(func(unflatten(_x), *args))[0]
    return _func, unflatten, _ex
