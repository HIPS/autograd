from __future__ import absolute_import
from autograd.core import (Node, FloatNode, VSpace, ComplexVSpace,
                           primitive, cast, getval,
                           register_node, register_vspace)
from . import numpy_wrapper as anp
from .numpy_extra import (ArrayNode, ArrayVSpace, array_dtype_mappings,
                          SparseArray, array_types)

class ComplexSparseArray(SparseArray):
    pass

class ComplexArrayVSpace(ArrayVSpace):
    def zeros(self):
        return anp.zeros(self.shape) + 0.0j

    def cast(self, value):
        result = complex_arraycast(value)
        if result.shape != self.shape:
            result = result.reshape(self.shape)
        return result

    @staticmethod
    def new_sparse_array(template, idx, x):
        return ComplexSparseArray(template, idx, x)

for complex_type in [anp.complex64, anp.complex128]:
    array_dtype_mappings[anp.dtype(complex_type)] = ComplexArrayVSpace
    register_node(FloatNode, complex_type)
    register_vspace(ComplexVSpace, complex_type)

@primitive
def complex_arraycast(val):
    return anp.array(val, dtype=complex)
complex_arraycast.defgrad(lambda g, ans, val: g)
