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
    @property
    def shape(self): return self.value.shape
    @property
    def ndim(self): return self.value.ndim
    @property
    def size(self): return self.value.size

    # Differentiable unary methods just apply to self
    squeeze = anp.squeeze
    ravel   = anp.ravel
    reshape = anp.reshape
    sum     = anp.sum
    mean    = anp.mean
    @property
    def T(self): return anp.transpose(self)
    __neg__ = P(op.neg)

    # Binary ops already wrapped by autograd.numpy.ndarray
    __add__  = anp.ndarray.__add__.__func__
    __sub__  = anp.ndarray.__sub__.__func__
    __mul__  = anp.ndarray.__mul__.__func__
    __pow__  = anp.ndarray.__pow__.__func__
    __div__  = anp.ndarray.__div__.__func__
    __radd__ = anp.ndarray.__radd__.__func__
    __rsub__ = anp.ndarray.__rsub__.__func__
    __rmul__ = anp.ndarray.__rmul__.__func__
    __rpow__ = anp.ndarray.__rpow__.__func__
    __rdiv__ = anp.ndarray.__rdiv__.__func__

Node.type_mappings[anp.ndarray] = ArrayNode
ArrayNode.__dict__['__neg__'].defgrad(lambda ans, x : op.neg)

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
