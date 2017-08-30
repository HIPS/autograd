"""
Handy functions for flattening nested containers containing numpy
arrays. The main purpose is to make examples and optimizers simpler.
"""
from autograd.tracer import getval
from autograd import make_vjp
import autograd.numpy as np

def flatten(value):
    """Flattens any nesting of tuples, lists, or dicts, with numpy arrays or
    scalars inside. Returns 1D numpy array and an unflatten function.
    Doesn't preserve mixed numeric types (e.g. floats and ints). Assumes dict
    keys are sortable."""
    unflatten, flat_value = make_vjp(_flatten)(value)
    return flat_value, unflatten

def _flatten(value):
    t = type(getval(value))
    if t in (list, tuple):
        return np.concatenate(map(_flatten, value))
    elif t is dict:
        return np.concatenate([_flatten(value[k]) for k in sorted(value)])
    else:
        return np.ravel(value)
