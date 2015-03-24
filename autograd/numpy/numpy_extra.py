from copy import copy
import operator as op
from autograd.core import Node, primitive as P, swap_args
from . import numpy_wrapper as anp

take = P(lambda A, idx : A[idx])
def make_grad_take(ans, A, idx):
    shape = A.shape
    return lambda g : untake(g, idx, shape)
take.defgrad(make_grad_take)

untake = P(lambda x, idx, shape : SparseArray(shape, idx, x))
untake.defgrad(lambda ans, x, idx, shape : lambda g : take(g, idx))

class ArrayNode(Node):
    __slots__ = []
    __getitem__ = take
    # Constants w.r.t float data just pass though
    shape = property(lambda self: self.value.shape)
    ndim  = property(lambda self: self.value.ndim)
    size  = property(lambda self: self.value.size)
    T = property(lambda self: anp.transpose(self))

Node.type_mappings[anp.ndarray] = ArrayNode
# Binary ops already wrapped by autograd.numpy.ndarray
inherited_methods = ['dot', '__neg__', '__add__',  '__sub__', '__mul__',
                     '__pow__', '__div__', '__radd__', '__rsub__',
                     '__rmul__', '__rpow__', '__rdiv__']
for method_name in inherited_methods:
    setattr(ArrayNode, method_name, getattr(anp.ndarray, method_name).__func__)

# These numpy.ndarray methods are just refs to an equivalent numpy function
nondiff_methods = ['all', 'any', 'argmax', 'argmin', 'argpartition',
                   'argsort', 'nonzero', 'searchsorted', 'round']
diff_methods = ['clip', 'compress', 'cumprod', 'cumsum', 'diagonal',
                'max', 'mean', 'min', 'prod', 'ptp', 'ravel', 'repeat',
                'reshape', 'squeeze', 'std', 'sum', 'swapaxes', 'take',
                'trace', 'transpose', 'var']
for method_name in nondiff_methods + diff_methods:
    setattr(ArrayNode, method_name, anp.__dict__[method_name])

# ----- Special sparse array type for efficient grads through indexing -----

class SparseArray(object):
    __array_priority__ = 150.0
    def __init__(self, shape, idx, val):
        self.shape = shape
        self.idx = idx
        self.val = val

    def __add__(self, other):
        array = anp.zeros(self.shape) if other is 0 else copy(other)
        array[self.idx] += self.val
        return array

    def __radd__(self, other):
        return self.__add__(other)

class SparseArrayNode(Node):
    __slots__ = []
    __add__  = P(SparseArray.__add__)
    __radd__ = P(SparseArray.__radd__)
Node.type_mappings[SparseArray] = SparseArrayNode

I = lambda x : x
SparseArrayNode.__dict__['__add__'].defgrad(lambda ans, x, y : I)
SparseArrayNode.__dict__['__add__'].defgrad(lambda ans, x, y : I, argnum=1)
SparseArrayNode.__dict__['__radd__'].grads = swap_args(SparseArrayNode.__dict__['__add__'].grads)
