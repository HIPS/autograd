from __future__ import absolute_import
from autograd.core import (Node, FloatNode, primitive,
                           differentiable_ops, nondifferentiable_ops, getval)
from . import numpy_wrapper as anp

np_float_types = [anp.float64, anp.float32, anp.float16]
for ft in np_float_types:
    Node.type_mappings[ft] = FloatNode

@primitive
def take(A, idx):
    return A[idx]
def make_grad_take(ans, A, idx):
    return lambda g : untake(g, idx, A)
take.defgrad(make_grad_take)

@primitive
def untake(x, idx, template):
    return SparseArray(template, idx, x)
untake.defgrad(lambda ans, x, idx, template : lambda g : take(g, idx))
untake.defgrad_is_zero(argnums=(1, 2))

Node.__array_priority__ = 90.0

class ArrayNode(Node):
    __slots__ = []
    __getitem__ = take
    __array_priority__ = 100.0

    # Constants w.r.t float data just pass though
    shape = property(lambda self: self.value.shape)
    ndim  = property(lambda self: self.value.ndim)
    size  = property(lambda self: self.value.size)
    T = property(lambda self: anp.transpose(self))

    @staticmethod
    def zeros_like(value):
        return anp.zeros(value.shape)

    @staticmethod
    def sum_outgrads(outgrads):
        if len(outgrads) is 1 and not isinstance(getval(outgrads[0]), SparseArray):
            return outgrads[0]
        else:
            return primitive_sum_arrays(*outgrads)

    def __neg__(self): return anp.negative(self)
    def __add__(self, other): return anp.add(     self, other)
    def __sub__(self, other): return anp.subtract(self, other)
    def __mul__(self, other): return anp.multiply(self, other)
    def __pow__(self, other): return anp.power   (self, other)
    def __div__(self, other): return anp.divide(  self, other)
    def __radd__(self, other): return anp.add(     other, self)
    def __rsub__(self, other): return anp.subtract(other, self)
    def __rmul__(self, other): return anp.multiply(other, self)
    def __rpow__(self, other): return anp.power   (other, self)
    def __rdiv__(self, other): return anp.divide(  other, self)
    def __eq__(self, other): return anp.equal(self, other)
    def __ne__(self, other): return anp.not_equal(self, other)
    def __gt__(self, other): return anp.greater(self, other)
    def __ge__(self, other): return anp.greater_equal(self, other)
    def __lt__(self, other): return anp.less(self, other)
    def __le__(self, other): return anp.less_equal(self, other)

Node.type_mappings[anp.ndarray] = ArrayNode

@primitive
def primitive_sum_arrays(*arrays):
    new_array = anp.zeros(arrays[0].shape)
    for array in arrays:
        if isinstance(array, SparseArray):
            new_array[array.idx] += array.val
        else:
            new_array += array
    return new_array
primitive_sum_arrays.gradmaker = lambda *args : lambda g : g

# These numpy.ndarray methods are just refs to an equivalent numpy function
nondiff_methods = ['all', 'any', 'argmax', 'argmin', 'argpartition',
                   'argsort', 'nonzero', 'searchsorted', 'round']
diff_methods = ['clip', 'compress', 'cumprod', 'cumsum', 'diagonal',
                'max', 'mean', 'min', 'prod', 'ptp', 'ravel', 'repeat',
                'reshape', 'squeeze', 'std', 'sum', 'swapaxes', 'take',
                'trace', 'transpose', 'var']
for method_name in nondiff_methods + diff_methods:
    setattr(ArrayNode, method_name, anp.__dict__[method_name])

# Replace FloatNode operators with broadcastable versions
for method_name in differentiable_ops + nondifferentiable_ops:
    setattr(FloatNode, method_name, ArrayNode.__dict__[method_name])

# ----- Special type for efficient grads through indexing -----

class SparseArray(object):
    __array_priority__ = 150.0
    def __init__(self, template, idx, val):
        self.shape = template.shape
        self.idx = idx
        self.val = val
Node.type_mappings[SparseArray] = ArrayNode
