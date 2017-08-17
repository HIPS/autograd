from __future__ import absolute_import
from autograd.core import (primitive, Box, VSpace, register_box, vspace,
                           register_vspace, SparseObject,
                           defvjp, defvjps, defvjp_is_zero, defvjp_argnum)
from builtins import zip
from future.utils import iteritems
from functools import partial
from .util import subvals
import autograd.numpy as np

class SequenceBox(Box):
    __slots__ = []
    def __getitem__(self, idx): return sequence_take(self, idx)
    def __len__(self): return len(self._value)
    def __add__(self, other): return sequence_extend_right(self, *other)
    def __radd__(self, other): return sequence_extend_left(self, *other)

register_box(SequenceBox, tuple)
register_box(SequenceBox, list)

@primitive
def sequence_take(A, idx):
    return A[idx]
def grad_sequence_take(ans, vs, gvs, A, idx):
    return lambda g: sequence_untake(g, idx, vs)
defvjp(sequence_take, grad_sequence_take)

@primitive
def sequence_extend_right(seq, *elts):
    return seq + type(seq)(elts)
def grad_sequence_extend_right(argnum, ans, vs, gvs, args, kwargs):
    seq, elts = args[0], args[1:]
    return lambda g: g[:len(seq)] if argnum == 0 else g[len(seq) + argnum - 1]
defvjp_argnum(sequence_extend_right, grad_sequence_extend_right)

@primitive
def sequence_extend_left(seq, *elts):
    return type(seq)(elts) + seq
def grad_sequence_extend_left(argnum, ans, vs, gvs, args, kwargs):
    seq, elts = args[0], args[1:]
    return lambda g: g[len(elts):] if argnum == 0 else g[argnum - 1]
defvjp_argnum(sequence_extend_left, grad_sequence_extend_left)

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
defvjp(sequence_untake, lambda ans, vs, gvs, x, idx, _: lambda g: sequence_take(g, idx))
defvjp_is_zero(sequence_untake, argnums=(1, 2))

@primitive
def make_sequence(seq_type, *args):
    return seq_type(args)
defvjp_argnum(make_sequence, lambda argnum, seq_type, *args: lambda g: g[argnum - 1])
make_tuple = partial(make_sequence, tuple)
make_list  = partial(make_sequence, list)

class SequenceVSpace(VSpace):
    def __init__(self, value):
        self.shape = [vspace(x) for x in value]
        self.size = sum(s.size for s in self.shape)
        self.seq_type = type(value)
        assert self.seq_type in (tuple, list)
    def zeros(self):
        return self.seq_type(vs.zeros() for vs in self.shape)
    def ones(self):
        return self.seq_type(vs.ones() for vs in self.shape)
    def standard_basis(self):
        zero = self.zeros()
        for i, vs in enumerate(self.shape):
            for x in vs.standard_basis():
                yield self.seq_type(subvals(zero, [(i, x)]))
    def randn(self):
        return self.seq_type(vs.randn() for vs in self.shape)
    def _add(self, xs, ys):
        return self.seq_type(vs._add(x, y) for x, y, vs in zip(xs, ys, self.shape))
    def _mut_add(self, xs, ys):
        return self.seq_type(vs._mut_add(x, y) for vs, x, y in zip(self.shape, xs, ys))
    def _scalar_mul(self, xs, a):
        return self.seq_type(vs._scalar_mul(x, a) for x, vs in zip(xs, self.shape))
    def _inner_prod(self, xs, ys):
        return sum(vs._inner_prod(x, y) for x, y, vs in zip(xs, ys, self.shape))
    def _covector(self, xs):
        return self.seq_type(vs._covector(x) for x, vs in zip(xs, self.shape))
register_vspace(SequenceVSpace, list)
register_vspace(SequenceVSpace, tuple)

class DictBox(Box):
    __slots__ = []
    def __getitem__(self, idx): return dict_take(self, idx)
    def __len__(self): return len(self._value)
    def __iter__(self): return self._value.__iter__()
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
defvjp(dict_take, grad_dict_take)

@primitive
def dict_untake(x, idx, vs):
    def mut_add(A):
         A[idx] = vs.shape[idx].mut_add(A[idx], x)
         return A
    return SparseObject(vs, mut_add)
defvjp(dict_untake, lambda ans, vs, gvs, x, idx, _: lambda g: dict_take(g, idx))
defvjp_is_zero(dict_untake, argnums=(1, 2))

def make_dict(pairs):
    keys, vals = zip(*pairs)
    return _make_dict(make_list(*keys), make_list(*vals))
@primitive
def _make_dict(keys, vals):
    return dict(zip(keys, vals))
defvjp(_make_dict, lambda ans, vs, gvs, keys, vals: lambda g:
       make_list(*[g[key] for key in keys]), argnum=1)

class DictVSpace(VSpace):
    def __init__(self, value):
        self.shape = {k : vspace(v) for k, v in iteritems(value)}
        self.size  = sum(s.size for s in self.shape.values())
    def zeros(self):
        return {k: vs.zeros()                  for k, vs in iteritems(self.shape)}
    def ones(self):
        return {k: vs.ones()                   for k, vs in iteritems(self.shape)}
    def standard_basis(self):
        zero = self.zeros()
        for k, vs in iteritems(self.shape):
            for x in vs.standard_basis():
                v = dict(iteritems(zero))
                v[k] = x
                yield v
    def randn(self):
        return {k: vs.randn()                  for k, vs in iteritems(self.shape)}
    def _add(self, xs, ys):
        return {k: vs._add(xs[k], ys[k])       for k, vs in iteritems(self.shape)}
    def _mut_add(self, xs, ys):
        return {k: vs._mut_add(xs[k], ys[k])   for k, vs in iteritems(self.shape)}
    def _scalar_mul(self, xs, a):
        return {k: vs._scalar_mul(xs[k], a)    for k, vs in iteritems(self.shape)}
    def _inner_prod(self, xs, ys):
        return sum(vs._inner_prod(xs[k], ys[k]) for k, vs in iteritems(self.shape))
    def _covector(self, xs):
        return {k: vs._covector(xs[k])         for k, vs in iteritems(self.shape)}
register_vspace(DictVSpace, dict)
