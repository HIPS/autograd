from __future__ import absolute_import
from autograd.core import primitive, Node, getval, zeros_like, cast

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
    def sum_outgrads(outgrads):
        return primitive_sum_tuples(*outgrads)

Node.type_mappings[tuple] = TupleNode

@primitive
def primitive_sum_tuples(*tuples):
    return tuple([sum(elements[1:], elements[0]) for elements in zip(*tuples)])
primitive_sum_tuples.gradmaker = lambda *args : lambda g : g

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
