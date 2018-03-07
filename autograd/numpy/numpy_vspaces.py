from collections import namedtuple
import numpy as np
from functools import reduce
import operator as op
from autograd.extend import VSpace, Box, primitive, defvjp, defjvp
from autograd.core import identity_jvp, identity_vjp, func
from autograd.builtins import isinstance as ag_isinstance

class ArrayVSpace(VSpace):
    def __init__(self, value):
        value = np.array(value, copy=False)
        self.shape = value.shape
        self.dtype = value.dtype

    @property
    def size(self): return np.prod(self.shape)
    @property
    def ndim(self): return len(self.shape)
    def zeros(self): return np.zeros(self.shape, dtype=self.dtype)
    def ones(self):  return np.ones( self.shape, dtype=self.dtype)

    def standard_basis(self):
      for idxs in np.ndindex(*self.shape):
          vect = np.zeros(self.shape, dtype=self.dtype)
          vect[idxs] = 1
          yield vect

    def randn(self):
        return np.array(np.random.randn(*self.shape)).astype(self.dtype)


    def densify(self, x):
        if x is None:
            return self.zeros()
        elif ag_isinstance(x, SparseArray):
            return self._sparse_to_dense(x)
        else:
            return x

    def _add(self, x, y):
        x_is_sparse = isinstance(x, SparseArray)
        y_is_sparse = isinstance(y, SparseArray)
        if x_is_sparse or y_is_sparse:
            x_sparse = x if x_is_sparse else SparseArray(self, x, [])
            y_sparse = y if y_is_sparse else SparseArray(self, y, [])
            arrays = filter(lambda a: a is not None, [x_sparse.array, y_sparse.array])
            new_array = reduce(op.add, arrays) if arrays else None
            return SparseArray(self, new_array, x_sparse.updates + y_sparse.updates)
        else:
            return x + y

    @primitive
    def _sparse_to_dense(self, x):
        out = self.zeros() if x.array is None else x.array.copy()
        for idx, val in x.updates:
            np.add.at(out, idx, val)
        return out

    def _inner_prod(self, x, y):
        return np.dot(np.ravel(x), np.ravel(y))

class ComplexArrayVSpace(ArrayVSpace):
    iscomplex = True

    @property
    def size(self): return np.prod(self.shape) * 2

    def ones(self):
        return (         np.ones(self.shape, dtype=self.dtype)
                + 1.0j * np.ones(self.shape, dtype=self.dtype))

    def standard_basis(self):
      for idxs in np.ndindex(*self.shape):
          for v in [1.0, 1.0j]:
              vect = np.zeros(self.shape, dtype=self.dtype)
              vect[idxs] = v
              yield vect

    def randn(self):
        return (         np.array(np.random.randn(*self.shape)).astype(self.dtype)
                + 1.0j * np.array(np.random.randn(*self.shape)).astype(self.dtype))

    def _inner_prod(self, x, y):
        return np.real(np.dot(np.conj(np.ravel(x)), np.ravel(y)))

    def _covector(self, x):
        return np.conj(x)

VSpace.register(np.ndarray,
                lambda x: ComplexArrayVSpace(x)
                if np.iscomplexobj(x)
                else ArrayVSpace(x))

for type_ in [float, np.float64, np.float32, np.float16]:
    ArrayVSpace.register(type_)

for type_ in [complex, np.complex64, np.complex128]:
    ComplexArrayVSpace.register(type_)

SparseArray = namedtuple("SparseArray", ["vs", "array", "updates"])
VSpace.register(SparseArray, lambda x : x.vs)
Box.register(SparseArray)
defvjp(func(ArrayVSpace._sparse_to_dense), None, identity_vjp)
defjvp(func(ArrayVSpace._sparse_to_dense), None, identity_jvp)
