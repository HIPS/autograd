import cupy as cp
import numpy as np
from autograd.extend import VSpace

class SparseArrayVSpace(VSpace):

    def __init__(self, value):
        self.t = type(value)
        self.value = self.t(value)
        self.shape = value.shape
        self.dtype = value.dtype

    @property
    def size(self):
        return self.value.size

    @property
    def ndim(self):
        return len(self.shape)

    def randn(self):
        a = cp.sparse.random(m=self.shape[0], n=self.shape[1])
        return self.t(a)

    def zeros(self):
        return self.t(self.shape)


VSpace.register(cp.sparse.csc_matrix, lambda x: SparseArrayVSpace(x))
VSpace.register(cp.sparse.csr_matrix, lambda x: SparseArrayVSpace(x))
VSpace.register(cp.sparse.coo_matrix, lambda x: SparseArrayVSpace(x))
VSpace.register(cp.sparse.dia_matrix, lambda x: SparseArrayVSpace(x))
