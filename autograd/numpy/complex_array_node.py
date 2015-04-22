from __future__ import absolute_import
from autograd.core import Node, ComplexNode, primitive, cast, getval
from . import numpy_wrapper as anp
from .numpy_extra import ArrayNode, array_dtype_mappings, SparseArray

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

    @staticmethod
    def new_sparse_array(template, idx, x):
        return ComplexSparseArray(template, idx, x)

for complex_type in [anp.complex64, anp.complex128]:
    array_dtype_mappings[anp.dtype(complex_type)] = ComplexArrayNode
    Node.type_mappings[complex_type] = ComplexNode

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

@primitive
def complex_arraycast(val):
    return anp.array(val, dtype=complex)
complex_arraycast.defgrad(lambda ans, val: lambda g : g)

class ComplexSparseArray(SparseArray):
    pass
Node.type_mappings[ComplexSparseArray] = ComplexArrayNode
