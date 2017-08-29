"""
Handy functions for flattening nested containers containing numpy
arrays. The main purpose is to make examples and optimizers simpler.
"""
from functools import partial
from autograd.tracer import notrace_primitive
import autograd.numpy as np

def flatten(value):
    shape = get_shape(value)
    return _flatten(shape, value), partial(_unflatten, shape)

def _flatten(shape, value):
    """Flattens any nesting of tuples, lists, or dicts, with numpy arrays or
    scalars inside. Returns 1D numpy array and an unflatten function.
    Doesn't preserve mixed numeric types (e.g. floats and ints). Assumes dict
    keys are sortable."""
    if shape.isleaf:
        return np.ravel(value)
    else:
        return np.concatenate(map(
            _flatten, shape.children, shape.child_values(value)))

def _unflatten(shape, value):
    if shape.isleaf:
        return np.reshape(value, shape.shape)
    else:
        split_children = split(value, [c.size for c in shape.children])
        return shape.new_container(map(_unflatten, shape.children, split_children))

def split(vector, sizes):
    return np.split(vector, np.cumsum(sizes[:-1]))

@notrace_primitive
def get_shape(value):
    return shape_types.get(type(value), Leaf)(value)

class ContainerShape(object):
    isleaf = False
    def __init__(self, value):
        self.children = map(get_shape, self.child_values(value))
        self.size = sum([c.size for c in self.children])
        self.type = type(value)

    def child_values(self, value):  return value
    def new_container(self, value): return self.type(value)

class DictShape(ContainerShape):
    def __init__(self, value):
        self.keys = sorted(value.keys())
        super(DictShape, self).__init__(value)

    def child_values(self, value):  return [value[k] for k in self.keys]
    def new_container(self, value): return {k: v for k, v in zip(self.keys, value)}

class Leaf(ContainerShape):
    isleaf = True
    def __init__(self, value):
        self.shape = np.shape(value)
        self.size  = np.size(value)

shape_types = {list:  ContainerShape, tuple: ContainerShape, dict:  DictShape}
