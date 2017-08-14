from __future__ import absolute_import
from autograd.core import (primitive, Box, VSpace, register_box, vspace,
                           register_vspace, SparseObject)
from builtins import zip
from future.utils import iteritems
from functools import partial
import autograd.numpy as np

class SequenceBox(Box):
    __slots__ = []
    def __getitem__(self, idx): return sequence_take(self, idx)
    def __len__(self): return len(self.value)
    def __add__(self, other): return sequence_extend_right(self, *other)
    def __radd__(self, other): return sequence_extend_left(self, *other)

register_box(SequenceBox, tuple)
register_box(SequenceBox, list)

@primitive
def sequence_take(A, idx):
    return A[idx]
def grad_sequence_take(ans, vs, gvs, A, idx):
    return lambda g: sequence_untake(g, idx, vs)
sequence_take.defvjp(grad_sequence_take)

@primitive
def sequence_extend_right(seq, *elts):
    return seq + type(seq)(elts)
def grad_sequence_extend_right(argnum, ans, vs, gvs, args, kwargs):
    seq, elts = args[0], args[1:]
    return lambda g: g[:len(seq)] if argnum == 0 else g[len(seq) + argnum - 1]
sequence_extend_right.vjp = grad_sequence_extend_right

@primitive
def sequence_extend_left(seq, *elts):
    return type(seq)(elts) + seq
def grad_sequence_extend_left(argnum, ans, vs, gvs, args, kwargs):
    seq, elts = args[0], args[1:]
    return lambda g: g[len(elts):] if argnum == 0 else g[argnum - 1]
sequence_extend_left.vjp = grad_sequence_extend_left

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
        return vs.seq_type(result)
    return SparseObject(vs, mut_add)
sequence_untake.defvjp(lambda ans, vs, gvs, x, idx, _: lambda g: sequence_take(g, idx))
sequence_untake.defvjp_is_zero(argnums=(1, 2))

@primitive
def make_sequence(seq_type, *args):
    return seq_type(args)
make_sequence.vjp = lambda argnum, seq_type, *args: lambda g: g[argnum - 1]
make_tuple = partial(make_sequence, tuple)
make_list  = partial(make_sequence, list)

class SequenceVSpace(VSpace):
    def __init__(self, value):
        self.shape = [vspace(x) for x in value]
        self.size = sum(s.size for s in self.shape)
        self.seq_type = type(value)
        assert self.seq_type in (tuple, list)

    def zeros(self):
        return self.seq_type(x.zeros() for x in self.shape)

    def add(self, x, y):
        return self.seq_type(vs.add(x, y) for x, y, vs in zip(x, y, self.shape))

    def scalar_mul(self, x, a):
        return self.seq_type(vs.scalar_mul(x, a) for x, vs in zip(x, self.shape))

    def inner_prod(self, x, y):
        return sum(vs.inner_prod(x, y) for x, y, vs in zip(x, y, self.shape))

    def randn(self):
        return self.seq_type(vs.randn() for vs in self.shape)

    def mut_add(self, xs, ys):
        return self.seq_type(vs.mut_add(x, y)
                                  for vs, x, y in zip(self.shape, xs, ys))

    def flatten(self, value, covector=False):
        if self.shape:
            return np.concatenate(
                [vs.flatten(v, covector) for vs, v in zip(self.shape, value)])
        else:
            return np.zeros((0,))

    def unflatten(self, value, covector=False):
        result = []
        start = 0
        for vs in self.shape:
            N = vs.size
            result.append(vs.unflatten(value[start:start + N], covector))
            start += N
        return self.seq_type(result)

register_vspace(SequenceVSpace, list)
register_vspace(SequenceVSpace, tuple)

class DictBox(Box):
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

register_box(DictBox, dict)

@primitive
def dict_take(A, idx):
    return A[idx]
def grad_dict_take(ans, vs, gvs, A, idx):
    return lambda g: dict_untake(g, idx, vs)
dict_take.defvjp(grad_dict_take)

@primitive
def dict_untake(x, idx, vs):
    def mut_add(A):
         A[idx] = vs.shape[idx].mut_add(A[idx], x)
         return A
    return SparseObject(vs, mut_add)
dict_untake.defvjp(lambda ans, vs, gvs, x, idx, _: lambda g: dict_take(g, idx))
dict_untake.defvjp_is_zero(argnums=(1, 2))

def make_dict(pairs):
    keys, vals = zip(*pairs)
    return _make_dict(make_list(*keys), make_list(*vals))
@primitive
def _make_dict(keys, vals):
    return dict(zip(keys, vals))
_make_dict.defvjp(lambda ans, vs, gvs, keys, vals: lambda g: make_list(*[g[key] for key in keys]),
                  argnum=1)

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
