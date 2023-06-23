import numpy as np
from autograd.extend import VSpace
from autograd.builtins import NamedTupleVSpace

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

for type_ in [float, np.longdouble, np.float64, np.float32, np.float16]:
    ArrayVSpace.register(type_)

for type_ in [complex, np.clongdouble, np.complex64, np.complex128]:
    ComplexArrayVSpace.register(type_)


if np.__version__ >= '1.25':
    class EigResultVSpace(NamedTupleVSpace):     seq_type = np.linalg.linalg.EigResult
    class EighResultVSpace(NamedTupleVSpace):    seq_type = np.linalg.linalg.EighResult
    class QRResultVSpace(NamedTupleVSpace):      seq_type = np.linalg.linalg.QRResult
    class SlogdetResultVSpace(NamedTupleVSpace): seq_type = np.linalg.linalg.SlogdetResult
    class SVDResultVSpace(NamedTupleVSpace):     seq_type = np.linalg.linalg.SVDResult

    EigResultVSpace.register(np.linalg.linalg.EigResult)
    EighResultVSpace.register(np.linalg.linalg.EighResult)
    QRResultVSpace.register(np.linalg.linalg.QRResult)
    SlogdetResultVSpace.register(np.linalg.linalg.SlogdetResult)
    SVDResultVSpace.register(np.linalg.linalg.SVDResult)
