from __future__ import absolute_import
from autograd.core import Node, ComplexNode, primitive, cast, getval
from . import numpy_wrapper as anp
from .numpy_extra import ArrayNode, array_dtype_mappings, SparseArray

class ComplexArrayNode(ArrayNode):
    @staticmethod
    def zeros_like(value):
        return anp.zeros(value.shape) + 0.0j

    @staticmethod
    def cast(value, example):
        result = complex_arraycast(value)
        if result.shape != example.shape:
            result = result.reshape(example.shape)
        return result

    @staticmethod
    def new_sparse_array(template, idx, x):
        return ComplexSparseArray(template, idx, x)

for complex_type in [anp.complex64, anp.complex128]:
    array_dtype_mappings[anp.dtype(complex_type)] = ComplexArrayNode
    Node.type_mappings[complex_type] = ComplexNode

@primitive
def complex_arraycast(val):
    return anp.array(val, dtype=complex)
complex_arraycast.defgrad(lambda ans, val: lambda g : g)

class ComplexSparseArray(SparseArray):
    pass
Node.type_mappings[ComplexSparseArray] = ComplexArrayNode
