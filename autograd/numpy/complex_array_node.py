from __future__ import absolute_import
from autograd.core import (Node, FloatNode, VSpace, ComplexVSpace,
                           primitive, cast, getval,
                           register_node, register_vspace)
from . import numpy_wrapper as anp
from .numpy_extra import ArrayNode, ArrayVSpace, array_dtype_mappings, array_types

for complex_type in [anp.complex64, anp.complex128]:
    register_node(FloatNode, complex_type)
    register_vspace(ComplexVSpace, complex_type)
