import numpy as np
from autograd.vspace import VSpace, register_vspace

class ArrayVSpace(VSpace):
    def __init__(self, value):
        value = np.array(value, copy=False)
        self.shape = value.shape
        self.dtype = value.dtype

    @property
    def size(self): return int(np.prod(self.shape))
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

    def _inner_prod(self, x, y):
        return np.dot(np.ravel(x), np.ravel(y))

    def _product(self, other_vspace):
        return self._contract(other_vspace, ndim=0)

    def _contract(self, other_vspace, ndim=None):
        ndim = other_vspace.ndim if ndim is None else ndim
        if not self.shape[-ndim % self.ndim:] == other_vspace.shape[:ndim]:
            raise ValueError

        result = self.__new__(self.__class__)
        result.shape = self.shape[:-ndim % self.ndim] + other_vspace.shape[ndim:]
        result.dtype = np.promote_types(self.dtype, other_vspace.dtype)
        return result

    def _kronecker_tensor(self):
        return np.reshape(np.eye(self.size), self.shape + self.shape)

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

register_vspace(lambda x: ComplexArrayVSpace(x)
                if np.iscomplexobj(x)
                else ArrayVSpace(x), np.ndarray)

for type_ in [float, np.float64, np.float32, np.float16]:
    register_vspace(ArrayVSpace, type_)

for type_ in [complex, np.complex64, np.complex128]:
    register_vspace(ComplexArrayVSpace, type_)
