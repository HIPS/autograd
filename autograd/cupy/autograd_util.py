import autograd.cupy as cupy
import autograd.core
from autograd.core import make_vjp, getval

def flatten(value):
    """Flattens any nesting of tuples, arrays, or dicts.
       Returns 1D array and an unflatten function.
       Doesn't preserve mixed numeric types (e.g. floats and ints)."""
    def flat(value):
        if isinstance(getval(value), cupy.ndarray):
            return cupy.ravel(value)
        elif isinstance(getval(value), (float, int)):
            return cupy.array([value])
        elif isinstance(getval(value), (tuple, list)):
            if not value: return array([])
            return cupy.concatenate(list(map(flat, value)))
        elif isinstance(getval(value), dict):
            items = sorted([(k, value[k]) for k in value], key=itemgetter(0))
            return cupy.concatenate([flat(val) for key, val in items])
        else:
            raise Exception("Don't know how to flatten type {}".format(type(value)))

    unflatten, node = make_vjp(flat)(value)
    unbox = not autograd.core.active_progenitors.intersection(node.progenitors)
    flattened_value = getval(node) if unbox else node
    return flattened_value, unflatten

def flatten_func(func, example):
    """Flattens both the inputs to a function, and the outputs."""
    flattened_example, unflatten = flatten(example)

    def flattened_func(flattened_params, *args, **kwargs):
        output = func(unflatten(flattened_params), *args, **kwargs)
        flattened_output, _ = flatten(output)
        return flattened_output
    return flattened_func, unflatten, flattened_example
