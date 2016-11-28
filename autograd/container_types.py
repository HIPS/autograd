from __future__ import absolute_import
from autograd.core import (primitive, Node, VSpace, register_node, vspace,
                           register_vspace, getval, SparseObject)
from builtins import zip
from future.utils import iteritems
import numpy as np

class TupleNode(Node):
    __slots__ = []
    def __getitem__(self, idx):
        return tuple_take(self, idx)
    def __len__(self):
        return len(self.value)

register_node(TupleNode, tuple)

@primitive
def tuple_take(A, idx):
    return A[idx]
def grad_tuple_take(g, ans, A, idx):
    return tuple_untake(g, idx, A)
tuple_take.defgrad(grad_tuple_take)

@primitive
def tuple_untake(x, idx, template):
    def mut_add(A):
        result = list(A)
        result[idx] = vs.shape[idx].mut_add(result[idx], x)
        return tuple(result)
    vs = vspace(template)
    return SparseObject(vs, mut_add)
tuple_untake.defgrad(lambda g, ans, x, idx, template : tuple_take(g, idx))
tuple_untake.defgrad_is_zero(argnums=(1, 2))

@primitive
def make_tuple(*args):
    return tuple(args)
make_tuple.grad = lambda argnum, g, *args: g[argnum]

class ListNode(Node):
    __slots__ = []
    def __getitem__(self, idx):
        return list_take(self, idx)
    def __len__(self):
        return len(self.value)

register_node(ListNode, list)

@primitive
def list_take(A, idx):
    return A[idx]
def grad_list_take(g, ans, A, idx):
    return list_untake(g, idx, A)
list_take.defgrad(grad_list_take)

@primitive
def list_untake(x, idx, template):
    def mut_add(A):
         A[idx] = vs.shape[idx].mut_add(A[idx], x)
         return A
    vs = vspace(template)
    return SparseObject(vs, mut_add)
list_untake.defgrad(lambda g, ans, x, idx, template : list_take(g, idx))
list_untake.defgrad_is_zero(argnums=(1, 2))

class DictNode(Node):
    __slots__ = []
    def __getitem__(self, idx):
        return dict_take(self, idx)
    def __len__(self):
        return len(self.value)
    def __iter__(self):
        return self.value.__iter__()

register_node(DictNode, dict)

@primitive
def dict_take(A, idx):
    return A[idx]
def grad_dict_take(g, ans, A, idx):
    return dict_untake(g, idx, A)
dict_take.defgrad(grad_dict_take)

@primitive
def dict_untake(x, idx, template):
    def mut_add(A):
         A[idx] = vs.shape[idx].mut_add(A[idx], x)
         return A
    vs = vspace(template)
    return SparseObject(vs, mut_add)
dict_untake.defgrad(lambda g, ans, x, idx, template : dict_take(g, idx))
dict_untake.defgrad_is_zero(argnums=(1, 2))

class SequenceVSpace(VSpace):
    def __init__(self, value):
        self.shape = [vspace(x) for x in value]
        self.size = sum(s.size for s in self.shape)
        self.sequence_type = type(value)
        assert self.sequence_type in (tuple, list)

    def zeros(self):
        return self.sequence_type(x.zeros() for x in self.shape)
    def mut_add(self, xs, ys):
        return self.sequence_type(vs.mut_add(x, y)
                                  for vs, x, y in zip(self.shape, xs, ys))
    def flatten(self, value):
        if self.shape:
            return np.concatenate(
                [s.flatten(v) for s, v in zip(self.shape, value)])
        else:
            return np.zeros((0,))

    def unflatten(self, value):
        result = []
        start = 0
        for s in self.shape:
            N = s.size

            result.append(s.unflatten(value[start:start + N]))
            start += N
        return self.sequence_type(result)

register_vspace(SequenceVSpace, list)
register_vspace(SequenceVSpace, tuple)

class DictVSpace(VSpace):
    def __init__(self, value):
        self.shape = {k : vspace(v) for k, v in value.iteritems()}
        self.size  = sum(s.size for s in self.shape.values())
    def zeros(self):
        return {k : v.zeros() for k, v in iteritems(self.shape)}
    def mut_add(self, xs, ys):
        return {k : v.mut_add(xs[k], ys[k])
                for k, v in self.shape.iteritems()}
    def flatten(self, value):
        if self.shape:
            return np.concatenate(
                [s.flatten(value[k])
                 for k, s in sorted(self.shape.iteritems())])
        else:
            return np.zeros((0,))

    def unflatten(self, value):
        result = {}
        start = 0
        for k, s in sorted(self.shape.iteritems()):
            N = s.size
            result[k] = s.unflatten(value[start:start + N])
            start += N
        return result

register_vspace(DictVSpace, dict)
