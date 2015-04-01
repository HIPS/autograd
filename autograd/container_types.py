from __future__ import absolute_import
from autograd.core import primitive, Node, getval, zeros_like

class TupleNode(Node):
    __slots__ = []
    def __getitem__(self, idx):
        return take(self, idx)
    def __len__(self):
        return len(self.value)

    @staticmethod
    def zeros_like(value):
        return tuple([zeros_like(item) for item in getval(value)])

    @staticmethod
    def iadd_any(A, B):
        if A is 0:
            return B
        else:
            return tuple([a + b for a, b in zip(A, B)])
Node.type_mappings[tuple] = TupleNode

@primitive
def take(A, idx):
    return A[idx]
def make_grad_take(ans, A, idx):
    return lambda g : untake(g, idx, A)
take.defgrad(make_grad_take)

@primitive
def untake(x, idx, template):
    result = list(zeros_like(template))
    result[idx] = x
    return tuple(result)
untake.defgrad(lambda ans, x, idx, template : lambda g : take(g, idx))
untake.defgrad_is_zero(argnums=(1, 2))

@primitive
def arg_tuple(*args):
    return tuple(args)
arg_tuple.gradmaker = lambda argnum, ans, *args : lambda g : g[argnum]
