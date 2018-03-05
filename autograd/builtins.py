from functools import partial
from .fmap_util import (container_fmap as fmap, fmap_to_list,
                        fmap_to_basis, fmap_to_zipped)
from .extend import notrace_primitive, VSpace, vspace

isinstance = notrace_primitive(isinstance)
type = notrace_primitive(type)
tuple, list, dict = tuple, list, dict

# This is now for test_util only. Could move fmaps in there.
class ContainerVSpace(VSpace):
    def __init__(self, value):
        self.shape = fmap(vspace, value)

    @property
    def size(self): return fsum(fmap(lambda vs: vs.size, self.shape))
    def zeros(self): return fmap(lambda vs: vs.zeros(), self.shape)
    def ones(self):  return fmap(lambda vs: vs.ones() , self.shape)
    def randn(self): return fmap(lambda vs: vs.randn(), self.shape)
    def standard_basis(self):
        zero = self.zeros()
        basis = fmap_to_basis(fmap, self.shape)
        for basis_elt, vs in fmap_to_zipped(fmap, basis, self.shape):
            for x in vs.standard_basis():
                yield fmap(lambda b, z: x if b else z, basis_elt.value, zero)

    def _add(self, xs, ys):
        return fmap(lambda vs, x, y: vs._add(x, y), self.shape, xs, ys)
    def _mut_add(self, xs, ys):
        return fmap(lambda vs, x, y: vs._mut_add(x, y), self.shape, xs, ys)
    def _scalar_mul(self, xs, a):
        return fmap(lambda vs, x: vs._scalar_mul(x, a), self.shape, xs)
    def _inner_prod(self, xs, ys):
        return fsum(fmap(lambda vs, x, y: vs._inner_prod(x, y), self.shape, xs, ys))
    def _covector(self, xs):
        return fmap(lambda vs, x: vs._covector(x), self.shape, xs)

for t in [list, tuple, dict]:
    ContainerVSpace.register(t)

def fsum(xs):
    return sum(fmap_to_list(fmap, xs))
