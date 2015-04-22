from __future__ import absolute_import
from autograd.core import Node, primitive, cast, getval
from . import numpy_wrapper as anp
from .numpy_extra import ArrayNode, array_dtype_mappings, SparseArray
from .use_gpu_numpy import use_gpu_numpy

assert use_gpu_numpy()

class GpuArrayNode(ArrayNode):
    @staticmethod
    def zeros_like(value):
        return anp.array(anp.zeros(value.shape), dtype=anp.gpu_float32)

    @staticmethod
    def cast(value):
        return gpu_arraycast(value)

    @staticmethod
    def new_sparse_array(template, idx, x):
        return GpuSparseArray(template, idx, x)
Node.type_mappings[anp.garray] = GpuArrayNode
array_dtype_mappings[anp.gpu_float32] = GpuArrayNode

@primitive
def gpu_arraycast(val):
    return anp.array(val, dtype=anp.gpu_float32)
gpu_arraycast.defgrad(lambda ans, val: lambda g : g)

class GpuSparseArray(SparseArray): pass
Node.type_mappings[GpuSparseArray] = GpuArrayNode
