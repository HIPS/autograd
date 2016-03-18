from __future__ import absolute_import
from autograd.core import (Node, ComplexNode, primitive, cast, getval,
                           register_node_type, type_mappings, return_this)
from . import numpy_wrapper as anp
from .numpy_extra import ArrayNode, array_dtype_mappings, SparseArray, array_types

class ComplexSparseArray(SparseArray):
    pass

class ComplexArrayNode(ArrayNode):
    value_types = [ComplexSparseArray]
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

register_node_type(ComplexArrayNode)
array_types.update([ComplexArrayNode, ComplexSparseArray])

for complex_type in [anp.complex64, anp.complex128]:
    array_dtype_mappings[anp.dtype(complex_type)] = ComplexArrayNode
    type_mappings[complex_type] = return_this(ComplexNode)

@primitive
def complex_arraycast(val):
    return anp.array(val, dtype=complex)
complex_arraycast.defgrad(lambda ans, val: lambda g : g)
