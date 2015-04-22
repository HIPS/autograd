from __future__ import absolute_import
from autograd.core import (Node, FloatNode, ComplexNode, primitive, cast,
                           differentiable_ops, nondifferentiable_ops, getval)
from . import numpy_wrapper as anp

for float_type in [anp.float64, anp.float32, anp.float16]:
    Node.type_mappings[float_type] = FloatNode
for complex_type in [anp.complex64, anp.complex128]:
    Node.type_mappings[complex_type] = ComplexNode

@primitive
def take(A, idx):
    return A[idx]
def make_grad_take(ans, A, idx):
    return lambda g : untake(g, idx, A)
take.defgrad(make_grad_take)

@primitive
def untake(x, idx, template):
    return new_sparse_array(template, idx, x)
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

    def __len__(self):
        return len(self.value)

    @staticmethod
    def zeros_like(value):
        return anp.zeros(value.shape)

    @staticmethod
    def sum_outgrads(outgrads):
        if len(outgrads) is 1 and not isinstance(getval(outgrads[0]), SparseArray):
            return outgrads[0]
        else:
            return primitive_sum_arrays(*outgrads)

    @staticmethod
    def cast(value):
        return arraycast(value)

    def __neg__(self): return anp.negative(self)
    def __add__(self, other): return anp.add(     self, other)
    def __sub__(self, other): return anp.subtract(self, other)
    def __mul__(self, other): return anp.multiply(self, other)
    def __pow__(self, other): return anp.power   (self, other)
    def __div__(self, other): return anp.divide(  self, other)
    def __mod__(self, other): return anp.mod(     self, other)
    def __radd__(self, other): return anp.add(     other, self)
    def __rsub__(self, other): return anp.subtract(other, self)
    def __rmul__(self, other): return anp.multiply(other, self)
    def __rpow__(self, other): return anp.power(   other, self)
    def __rdiv__(self, other): return anp.divide(  other, self)
    def __rmod__(self, other): return anp.mod(     other, self)
    def __eq__(self, other): return anp.equal(self, other)
    def __ne__(self, other): return anp.not_equal(self, other)
    def __gt__(self, other): return anp.greater(self, other)
    def __ge__(self, other): return anp.greater_equal(self, other)
    def __lt__(self, other): return anp.less(self, other)
    def __le__(self, other): return anp.less_equal(self, other)

class ComplexArrayNode(ArrayNode):
    @staticmethod
    def zeros_like(value):
        return anp.zeros(value.shape) + 0.0j

    @staticmethod
    def sum_outgrads(outgrads):
        if len(outgrads) is 1 and not isinstance(getval(outgrads[0]), SparseArray):
            return outgrads[0]
        else:
            return primitive_sum_arrays_complex(*outgrads)

    @staticmethod
    def cast(value):
        return complex_arraycast(value)

def new_array_node(value, tapes):
    if anp.iscomplexobj(value):
        return ComplexArrayNode(value, tapes)
    else:
        return ArrayNode(value, tapes)

Node.type_mappings[anp.ndarray] = new_array_node

@primitive
def arraycast(val):
    if isinstance(val, float):
        return anp.array(val)
    elif anp.iscomplexobj(val):
        return anp.array(anp.real(val))
    else:
        raise TypeError("Can't cast type {0} to array".format(type(val)))
arraycast.defgrad(lambda ans, val: lambda g : g)

@primitive
def complex_arraycast(val):
    return anp.array(val, dtype=complex)
complex_arraycast.defgrad(lambda ans, val: lambda g : g)

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

@primitive
def primitive_sum_arrays_complex(*arrays):
    new_array = anp.zeros(arrays[0].shape).astype(complex)
    for array in arrays:
        if isinstance(array, SparseArray):
            new_array[array.idx] += array.val
        else:
            new_array += array
    return new_array
primitive_sum_arrays_complex.gradmaker = lambda *args : lambda g : g

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

class ComplexSparseArray(SparseArray):
    pass
Node.type_mappings[ComplexSparseArray] = ComplexArrayNode

def new_sparse_array(template, idx, val):
    if anp.iscomplexobj(template):
        return ComplexSparseArray(template, idx, val)
    else:
        return SparseArray(template, idx, val)
