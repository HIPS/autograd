from __future__ import absolute_import
import numpy as np

from autograd.core import (Node, FloatNode, VSpace, FloatVSpace, primitive, cast,
                           register_node, register_vspace, zeros_like,
                           differentiable_ops, nondifferentiable_ops, getval)
from . import numpy_wrapper as anp

@primitive
def take(A, idx):
    return A[idx]
def grad_take(g, ans, A, idx):
    return untake(g, idx, A)
take.defgrad(grad_take)

@primitive
def untake(x, idx, template):
    return array_dtype_mappings[template.dtype].new_sparse_array(template, idx, x)
untake.defgrad(lambda g, ans, x, idx, template : take(g, idx))
untake.defgrad_is_zero(argnums=(1, 2))

Node.__array_priority__ = 90.0

class SparseArray(object):
    # Special type for efficient grads through indexing
    __array_priority__ = 150.0
    def __init__(self, template, idx, val):
        self.shape = template.shape
        self.idx = idx
        self.val = val
        self.dtype = template.dtype

class ArrayNode(Node):
    __slots__ = []
    __getitem__ = take
    __array_priority__ = 100.0

    # Constants w.r.t float data just pass though
    shape = property(lambda self: self.value.shape)
    ndim  = property(lambda self: self.value.ndim)
    size  = property(lambda self: self.value.size)
    dtype = property(lambda self: self.value.dtype)
    T = property(lambda self: anp.transpose(self))

    def __len__(self):
        return len(self.value)

    def __neg__(self): return anp.negative(self)
    def __add__(self, other): return anp.add(     self, other)
    def __sub__(self, other): return anp.subtract(self, other)
    def __mul__(self, other): return anp.multiply(self, other)
    def __pow__(self, other): return anp.power   (self, other)
    def __div__(self, other): return anp.divide(  self, other)
    def __mod__(self, other): return anp.mod(     self, other)
    def __truediv__(self, other): return anp.true_divide(self, other)
    def __radd__(self, other): return anp.add(     other, self)
    def __rsub__(self, other): return anp.subtract(other, self)
    def __rmul__(self, other): return anp.multiply(other, self)
    def __rpow__(self, other): return anp.power(   other, self)
    def __rdiv__(self, other): return anp.divide(  other, self)
    def __rmod__(self, other): return anp.mod(     other, self)
    def __rtruediv__(self, other): return anp.true_divide(other, self)
    def __eq__(self, other): return anp.equal(self, other)
    def __ne__(self, other): return anp.not_equal(self, other)
    def __gt__(self, other): return anp.greater(self, other)
    def __ge__(self, other): return anp.greater_equal(self, other)
    def __lt__(self, other): return anp.less(self, other)
    def __le__(self, other): return anp.less_equal(self, other)

class ArrayVSpace(VSpace):
    def __init__(self, value):
        self.shape = value.shape

    def zeros(self):
        return anp.zeros(self.shape)

    def sum_outgrads(self, outgrads):
        if len(outgrads) is 1 and not isinstance(getval(outgrads[0]), SparseArray):
            return outgrads[0]
        else:
            return primitive_sum_arrays(*outgrads)

    def cast(self, value):
        result = arraycast(value)
        if result.shape != self.shape:
            result = result.reshape(self.shape)
        return result

    @staticmethod
    def new_sparse_array(template, idx, x):
        return SparseArray(template, idx, x)

def array_vspace(value):
    try:
        return array_dtype_mappings[value.dtype](value)
    except KeyError:
        raise TypeError("Can't differentiate wrt numpy arrays of dtype {0}".format(value.dtype))

register_node(ArrayNode, np.ndarray)
register_node(ArrayNode, SparseArray)
register_vspace(array_vspace, np.ndarray)
register_vspace(array_vspace, SparseArray)

array_types = set([anp.ndarray, SparseArray, ArrayNode])

array_dtype_mappings = {}
for float_type in [anp.float64, anp.float32, anp.float16]:
    array_dtype_mappings[anp.dtype(float_type)] = ArrayVSpace
    register_node(FloatNode, float_type)
    register_vspace(FloatVSpace, float_type)

@primitive
def arraycast(val):
    if isinstance(val, float):
        return anp.array(val)
    elif anp.iscomplexobj(val):
        return anp.array(anp.real(val))
    else:
        raise TypeError("Can't cast type {0} to array".format(type(val)))
arraycast.defgrad(lambda g, ans, val: g)

@primitive
def primitive_sum_arrays(*arrays):
    new_array = zeros_like(arrays[0]) # TODO: simplify this
    for array in arrays:
        if isinstance(array, SparseArray):
            np.add.at(new_array, array.idx, array.val)
        else:
            new_array += array
    return new_array
primitive_sum_arrays.grad = lambda argnum, g, *args : g

# These numpy.ndarray methods are just refs to an equivalent numpy function
nondiff_methods = ['all', 'any', 'argmax', 'argmin', 'argpartition',
                   'argsort', 'nonzero', 'searchsorted', 'round']
diff_methods = ['clip', 'compress', 'cumprod', 'cumsum', 'diagonal',
                'max', 'mean', 'min', 'prod', 'ptp', 'ravel', 'repeat',
                'reshape', 'squeeze', 'std', 'sum', 'swapaxes', 'take',
                'trace', 'transpose', 'var']
for method_name in nondiff_methods + diff_methods:
    setattr(ArrayNode, method_name, anp.__dict__[method_name])

# Flatten has no function, only a method.
setattr(ArrayNode, 'flatten', anp.__dict__['ravel'])

# Replace FloatNode operators with broadcastable versions
for method_name in differentiable_ops + nondifferentiable_ops + ['__truediv__', '__rtruediv__']:
    setattr(FloatNode, method_name, ArrayNode.__dict__[method_name])
