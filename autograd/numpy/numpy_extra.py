from __future__ import absolute_import
from copy import copy
import numpy as numpy_original
from autograd.core import (Node, primitive, zeros_like,
                           differentiable_ops, nondifferentiable_ops)
from . import numpy_wrapper as anp

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

class ArrayNode(Node):
    __slots__ = []
    __getitem__ = take
    __array_priority__ = 100.0

    def __init__(self, value):
        if type(Node) is numpy_original.ndarray:
            value = value.view(anp.ndarray)
        self.value = value
        self.tapes = {}

    # Constants w.r.t float data just pass though
    shape = property(lambda self: self.value.shape)
    ndim  = property(lambda self: self.value.ndim)
    size  = property(lambda self: self.value.size)
    T = property(lambda self: anp.transpose(self))

    @staticmethod
    def zeros_like(value):
        return anp.zeros(value.shape)

Node.type_mappings[anp.ndarray] = ArrayNode
Node.type_mappings[numpy_original.ndarray] = ArrayNode

# Binary ops already wrapped by autograd.numpy.ndarray
inherited_methods = ['dot'] + differentiable_ops +  nondifferentiable_ops
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


# ----- Special type for efficient grads through indexing -----

class SparseArray(object):
    __array_priority__ = 150.0
    def __init__(self, template, idx, val):
        self.template = template
        self.idx = idx
        self.val = val

class SparseArrayNode(Node):
    @staticmethod
    def iadd_any(other, self):
        array = zeros_like(self.template) if other is 0 else other
        array[self.idx] += self.val
        return array
Node.type_mappings[SparseArray] = SparseArrayNode
