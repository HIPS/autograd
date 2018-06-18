import cupy as cp
import numpy as np
from autograd.extend import VSpace


class ArrayVSpace(VSpace):

    def __init__(self, value):
        value = cp.array(value, copy=False)
        self.shape = value.shape
        self.dtype = value.dtype

    @property
    def size(self):
        return cp.prod(cp.array(self.shape))

    @property
    def ndim(self):
        return len(self.shape)

    def zeros(self):
        return cp.zeros(self.shape, dtype=self.dtype)

    def ones(self):
        return cp.ones(self.shape, dtype=self.dtype)

    def standard_basis(self):
        for idxs in cp.ndindex(*self.shape):
            vect = cp.zeros(self.shape, dtype=self.dtype)
            vect[idxs] = 1
            yield vect

    def randn(self):
        return cp.array(cp.random.randn(*self.shape)).astype(self.dtype)

    def _inner_prod(self, x, y):
        return cp.dot(cp.ravel(x), cp.ravel(y))


class ComplexArrayVSpace(ArrayVSpace):
    iscomplex = True

    @property
    def size(self):
        return cp.prod(self.shape) * 2

    def ones(self):
        return (
            cp.ones(self.shape, dtype=self.dtype)
            + 1.0j
            * cp.ones(self.shape, dtype=self.dtype)
        )

    def standard_basis(self):
        for idxs in cp.ndindex(*self.shape):
            for v in [1.0, 1.0j]:
                vect = cp.zeros(self.shape, dtype=self.dtype)
                vect[idxs] = v
                yield vect

    def randn(self):
        return (
            cp.array(cp.random.randn(*self.shape)).astype(self.dtype)
            + 1.0j
            * cp.array(cp.random.randn(*self.shape)).astype(self.dtype)
        )

    def _inner_prod(self, x, y):
        return cp.real(cp.dot(cp.conj(cp.ravel(x)), cp.ravel(y)))

    def _covector(self, x):
        return cp.conj(x)


VSpace.register(
    cp.ndarray,
    lambda x: ComplexArrayVSpace(x) if cp.iscomplexobj(x) else ArrayVSpace(x),
)

float_types = [float, cp.float64, cp.float32, cp.float16]

for type_ in float_types:
    ArrayVSpace.register(type_)

complex_types = [complex, np.complex64, np.complex128]

for type_ in complex_types:
    ComplexArrayVSpace.register(type_)
