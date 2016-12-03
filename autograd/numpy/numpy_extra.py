from __future__ import absolute_import
import numpy as np

from autograd.core import (Node, VSpace,
                           SparseObject, primitive, vspace,
                           register_node, register_vspace, getval)
from . import numpy_wrapper as anp

@primitive
def take(A, idx):
    return A[idx]
def grad_take(g, ans, vs, gvs, A, idx):
    return untake(g, idx, A)
take.defvjp(grad_take)

@primitive
def untake(x, idx, template):
    def mut_add(A):
        np.add.at(A, idx, x)
        return A
    return SparseObject(vspace(template), mut_add)
untake.defvjp(lambda g, ans, vs, gvs, x, idx, template : take(g, idx))
untake.defvjp_is_zero(argnums=(1, 2))

Node.__array_priority__ = 90.0

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
    def __matmul__(self, other): return anp.matmul(self, other)
    def __radd__(self, other): return anp.add(     other, self)
    def __rsub__(self, other): return anp.subtract(other, self)
    def __rmul__(self, other): return anp.multiply(other, self)
    def __rpow__(self, other): return anp.power(   other, self)
    def __rdiv__(self, other): return anp.divide(  other, self)
    def __rmod__(self, other): return anp.mod(     other, self)
    def __rtruediv__(self, other): return anp.true_divide(other, self)
    def __rmatmul__(self, other): return anp.matmul(other, self)
    def __eq__(self, other): return anp.equal(self, other)
    def __ne__(self, other): return anp.not_equal(self, other)
    def __gt__(self, other): return anp.greater(self, other)
    def __ge__(self, other): return anp.greater_equal(self, other)
    def __lt__(self, other): return anp.less(self, other)
    def __le__(self, other): return anp.less_equal(self, other)
    def __abs__(self): return anp.abs(self)
    def __hash__(self): return id(self)

class ArrayVSpace(VSpace):
    def __init__(self, value):
        value = np.array(value)
        self.shape = value.shape
        self.size  = value.size
        self.dtype = value.dtype
        self.scalartype = float

    def zeros(self):
        return anp.zeros(self.shape, dtype=self.dtype)

    def flatten(self, value, covector=False):
        return np.ravel(value)

    def unflatten(self, value, covector=False):
        return value.reshape(self.shape)

    def examples(self):
        # many possible instantiations
        original_examples = super(ArrayVSpace, self).examples()
        if self.shape == ():
            np_scalar_examples = [ex[()] for ex in original_examples]
            py_scalar_examples = list(map(self.scalartype, np_scalar_examples))
            return original_examples + np_scalar_examples + py_scalar_examples
        else:
            return original_examples

class ComplexArrayVSpace(ArrayVSpace):
    iscomplex = True
    def __init__(self, value):
        super(ComplexArrayVSpace, self).__init__(value)
        self.size  = 2 * self.size
        self.scalartype = complex

    def flatten(self, value, covector=False):
        if covector:
            return np.ravel(np.stack([np.real(value), - np.imag(value)]))
        else:
            return np.ravel(np.stack([np.real(value), np.imag(value)]))

    def unflatten(self, value, covector=False):
        reshaped = np.reshape(value, (2,) + self.shape)
        if covector:
            return np.array(reshaped[0] - 1j * reshaped[1])
        else:
            return np.array(reshaped[0] + 1j * reshaped[1])

register_node(ArrayNode, np.ndarray)
register_vspace(lambda x: ComplexArrayVSpace(x)
                if np.iscomplexobj(x)
                else ArrayVSpace(x), np.ndarray)
array_types = set([anp.ndarray, ArrayNode])

for type_ in [float, anp.float64, anp.float32, anp.float16]:
    register_node(ArrayNode, type_)
    register_vspace(ArrayVSpace, type_)

for type_ in [complex, anp.complex64, anp.complex128]:
    register_node(ArrayNode, type_)
    register_vspace(ComplexArrayVSpace, type_)

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
