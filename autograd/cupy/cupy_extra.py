from __future__ import absolute_import
import cupy

from autograd.core import (Node, VSpace, SparseObject, primitive, vspace,
                           register_node, register_vspace, getval)

from . import cupy_wrapper as acp

@primitive
def take(A, idx):
    # NOTE(mattjj): cupy doesn't support lists of slices
    return A[idx]
take.defvjp(lambda g, ans, vs, gvs, A, idx: untake(g, idx, A))

@primitive
def take_axis(A, idx, axis=0):
    return cupy.take(A, idx, axis)
def take_axis_grad(g, ans, vs, gvs, A, idx, axis=0):
    slices = [slice(None)] * ans.ndim
    slices[axis] = idx
    return untake(g, slices, A)
take_axis.defvjp(take_axis_grad)

@primitive
def untake(x, idx, template):
    def mut_add(A):
        cupy.scatter_add(A, idx, x)
        return A
    return SparseOjbect(vspace(template), mut_add)
untake.defvjp(lambda g, ans, vs, gvs, x, idx, template: take(g, idx))
untake.defvjp_is_zero(argnums=(1, 2))

class CupyArrayNode(Node):
    __slots__ = []
    __getitem__ = take
    __array_priority__ = 110.0  # escalation...

    shape = property(lambda self: self.value.shape)
    ndim  = property(lambda self: self.value.ndim)
    size  = property(lambda self: self.value.size)
    dtype = property(lambda self: self.value.dtype)
    T = property(lambda self: acp.transpose(self))

    def get(self): return acp.asnumpy(self)

    def __len__(self): return len(self.value)

    def __neg__(self): return acp.negative(self)
    def __add__(self, other): return acp.add(     self, other)
    def __sub__(self, other): return acp.subtract(self, other)
    def __mul__(self, other): return acp.multiply(self, other)
    def __pow__(self, other): return acp.power   (self, other)
    def __div__(self, other): return acp.divide(  self, other)
    def __mod__(self, other): return acp.mod(     self, other)
    def __truediv__(self, other): return acp.true_divide(self, other)
    def __matmul__(self, other): return acp.matmul(self, other)
    def __radd__(self, other): return acp.add(     other, self)
    def __rsub__(self, other): return acp.subtract(other, self)
    def __rmul__(self, other): return acp.multiply(other, self)
    def __rpow__(self, other): return acp.power(   other, self)
    def __rdiv__(self, other): return acp.divide(  other, self)
    def __rmod__(self, other): return acp.mod(     other, self)
    def __rtruediv__(self, other): return acp.true_divide(other, self)
    def __rmatmul__(self, other): return acp.matmul(other, self)
    def __eq__(self, other): return acp.equal(self, other)
    def __ne__(self, other): return acp.not_equal(self, other)
    def __gt__(self, other): return acp.greater(self, other)
    def __ge__(self, other): return acp.greater_equal(self, other)
    def __lt__(self, other): return acp.less(self, other)
    def __le__(self, other): return acp.less_equal(self, other)
    def __abs__(self): return acp.abs(self)
    def __hash__(self): return id(self)

class CupyArrayVSpace(VSpace):
    def __init__(self, value):
        value = cupy.array(value, copy=False)
        self.shape = value.shape
        self.size  = value.size
        self.dtype = value.dtype
        self.scalartype = float

    def zeros(self):
        return acp.zeros(self.shape, dtype=self.dtype)

    def flatten(self, value, covector=False):
        return acp.ravel(value)

    def unflatten(self, value, covector=False):
        return value.reshape(self.shape)

    def examples(self):
        raise NotImplementedError

register_node(CupyArrayNode, cupy.ndarray)
register_vspace(lambda x: CupyArrayVSpace(x), cupy.ndarray)

for type_ in [float, cupy.float16, cupy.float32, cupy.float64]:
    register_node(CupyArrayNode, type_)
    register_vspace(CupyArrayVSpace, type_)

# These cupy.ndarray methods are just refs to an equivalent numpy function
nondiff_methods = ['all', 'any', 'argmax', 'argmin', 'nonzero']
diff_methods = ['clip', 'cumsum', 'diagonal', 'max', 'mean', 'min', 'prod',
                'ravel', 'repeat', 'reshape', 'squeeze', 'std', 'sum',
                'swapaxes', 'take', 'trace', 'transpose', 'var']
for method_name in nondiff_methods + diff_methods:
    setattr(CupyArrayNode, method_name, acp.__dict__[method_name])

setattr(CupyArrayNode, 'flatten', acp.__dict__['ravel'])
