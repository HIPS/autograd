from __future__ import absolute_import
from autograd.core import (primitive, Node, VSpace, register_node, vspace,
                           register_vspace, getval, SparseObject)
from builtins import zip
from future.utils import iteritems
from functools import partial
import numpy as np

class SequenceNode(Node):
    __slots__ = []
    def __getitem__(self, idx):
        return sequence_take(self, idx)
    def __len__(self):
        return len(self.value)

register_node(SequenceNode, tuple)
register_node(SequenceNode, list)

@primitive
def sequence_take(A, idx):
    return A[idx]
def grad_sequence_take(g, ans, vs, gvs, A, idx):
    return sequence_untake(g, idx, vspace(getval(A)))
sequence_take.defvjp(grad_sequence_take)

@primitive
def sequence_untake(x, idx, vs):
    if isinstance(idx, int):
        accum = lambda result: vs.shape[idx].mut_add(result, x)
    else:
        accum = lambda result: [elt_vs.mut_add(a, b)
                                for elt_vs, a, b in zip(vs.shape[idx], result, x)]
    def mut_add(A):
        result = list(A)
        result[idx] = accum(result[idx])
        return vs.sequence_type(result)
    return SparseObject(vs, mut_add)
sequence_untake.defvjp(lambda g, ans, vs, gvs, x, idx, template : sequence_take(g, idx))
sequence_untake.defvjp_is_zero(argnums=(1, 2))

@primitive
def make_sequence(sequence_type, *args):
    return sequence_type(args)
make_sequence.vjp = lambda argnum, g, sequence_type, *args: g[argnum - 1]
make_tuple = partial(make_sequence, tuple)
make_list  = partial(make_sequence, list)

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

    def flatten(self, value, covector=False):
        if self.shape:
            return np.concatenate(
                [s.flatten(v, covector) for s, v in zip(self.shape, value)])
        else:
            return np.zeros((0,))

    def unflatten(self, value, covector=False):
        result = []
        start = 0
        for s in self.shape:
            N = s.size

            result.append(s.unflatten(value[start:start + N], covector))
            start += N
        return self.sequence_type(result)

register_vspace(SequenceVSpace, list)
register_vspace(SequenceVSpace, tuple)

class DictNode(Node):
    __slots__ = []
    def __getitem__(self, idx): return dict_take(self, idx)
    def __len__(self): return len(self.value)
    def __iter__(self): return self.value.__iter__()
    def items(self): return list(self.iteritems())
    def keys(self): return list(self.iterkeys())
    def values(self): return list(self.itervalues())
    def iteritems(self): return ((k, self[k]) for k in self)
    def iterkeys(self): return iter(self)
    def itervalues(self): return (self[k] for k in self)

register_node(DictNode, dict)

@primitive
def dict_take(A, idx):
    return A[idx]
def grad_dict_take(g, ans, vs, gvs, A, idx):
    return dict_untake(g, idx, A)
dict_take.defvjp(grad_dict_take)

@primitive
def dict_untake(x, idx, template):
    def mut_add(A):
         A[idx] = vs.shape[idx].mut_add(A[idx], x)
         return A
    vs = vspace(template)
    return SparseObject(vs, mut_add)
dict_untake.defvjp(lambda g, ans, vs, gvs, x, idx, template : dict_take(g, idx))
dict_untake.defvjp_is_zero(argnums=(1, 2))

class DictVSpace(VSpace):
    def __init__(self, value):
        self.shape = {k : vspace(v) for k, v in iteritems(value)}
        self.size  = sum(s.size for s in self.shape.values())
    def zeros(self):
        return {k : v.zeros() for k, v in iteritems(self.shape)}
    def mut_add(self, xs, ys):
        return {k : v.mut_add(xs[k], ys[k])
                for k, v in iteritems(self.shape)}
    def flatten(self, value, covector=False):
        if self.shape:
            return np.concatenate(
                [s.flatten(value[k], covector)
                 for k, s in sorted(iteritems(self.shape))])
        else:
            return np.zeros((0,))

    def unflatten(self, value, covector=False):
        result = {}
        start = 0
        for k, s in sorted(iteritems(self.shape)):
            N = s.size
            result[k] = s.unflatten(value[start:start + N], covector)
            start += N
        return result

register_vspace(DictVSpace, dict)
