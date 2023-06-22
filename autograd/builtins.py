import itertools
from future.utils import with_metaclass
from .util import subvals
from .extend import (Box, primitive, notrace_primitive, VSpace, vspace,
                     SparseObject, defvjp, defvjp_argnum, defjvp, defjvp_argnum)

isinstance_ = isinstance
isinstance = notrace_primitive(isinstance)

type_ = type
type = notrace_primitive(type)

tuple_, list_, dict_ = tuple, list, dict

@primitive
def container_take(A, idx):
    return A[idx]
def grad_container_take(ans, A, idx):
    return lambda g: container_untake(g, idx, vspace(A))
defvjp(container_take, grad_container_take)
defjvp(container_take, 'same')

class SequenceBox(Box):
    __slots__ = []
    __getitem__ = container_take
    def __len__(self): return len(self._value)
    def __add__(self, other): return sequence_extend_right(self, *other)
    def __radd__(self, other): return sequence_extend_left(self, *other)
    def __contains__(self, elt): return elt in self._value
    def index(self, elt): return self._value.index(elt)
SequenceBox.register(tuple_)
SequenceBox.register(list_)

class DictBox(Box):
    __slots__ = []
    __getitem__= container_take
    def __len__(self): return len(self._value)
    def __iter__(self): return self._value.__iter__()
    def __contains__(self, elt): return elt in self._value
    def items(self): return list(self.iteritems())
    def keys(self): return list(self.iterkeys())
    def values(self): return list(self.itervalues())
    def iteritems(self): return ((k, self[k]) for k in self)
    def iterkeys(self): return iter(self)
    def itervalues(self): return (self[k] for k in self)
    def get(self, k, d=None): return self[k] if k in self else d
DictBox.register(dict_)

@primitive
def container_untake(x, idx, vs):
    if isinstance(idx, slice):
        accum = lambda result: [elt_vs._mut_add(a, b)
                                for elt_vs, a, b in zip(vs.shape[idx], result, x)]
    else:
        accum = lambda result: vs.shape[idx]._mut_add(result, x)
    def mut_add(A):
        return vs._subval(A, idx, accum(A[idx]))
    return SparseObject(vs, mut_add)
defvjp(container_untake, lambda ans, x, idx, _:
       lambda g: container_take(g, idx))
defjvp(container_untake, 'same')

@primitive
def sequence_extend_right(seq, *elts):
    return seq + type(seq)(elts)
def grad_sequence_extend_right(argnum, ans, args, kwargs):
    seq, elts = args[0], args[1:]
    return lambda g: g[:len(seq)] if argnum == 0 else g[len(seq) + argnum - 1]
defvjp_argnum(sequence_extend_right, grad_sequence_extend_right)

@primitive
def sequence_extend_left(seq, *elts):
    return type(seq)(elts) + seq
def grad_sequence_extend_left(argnum, ans, args, kwargs):
    seq, elts = args[0], args[1:]
    return lambda g: g[len(elts):] if argnum == 0 else g[argnum - 1]
defvjp_argnum(sequence_extend_left, grad_sequence_extend_left)

@primitive
def make_sequence(seq_type, *args):
    return seq_type(args)
defvjp_argnum(make_sequence, lambda argnum, *args: lambda g: g[argnum - 1])

def fwd_grad_make_sequence(argnum, g, ans, seq_type, *args, **kwargs):
    return container_untake(g, argnum-1, vspace(ans))

defjvp_argnum(make_sequence, fwd_grad_make_sequence)


class TupleMeta(type_):
    def __instancecheck__(self, instance):
        return isinstance(instance, tuple_)
class tuple(with_metaclass(TupleMeta, tuple_)):
    def __new__(cls, xs):
        return make_sequence(tuple_, *xs)

class ListMeta(type_):
    def __instancecheck__(self, instance):
        return isinstance(instance, list_)
class list(with_metaclass(ListMeta, list_)):
    def __new__(cls, xs):
        return make_sequence(list_,  *xs)

class DictMeta(type_):
    def __instancecheck__(self, instance):
        return isinstance(instance, dict_)
class dict(with_metaclass(DictMeta, dict_)):
    def __new__(cls, *args, **kwargs):
        result = dict_(*args, **kwargs)
        if result:
            return _make_dict(result.keys(), list(result.values()))
        return result

@primitive
def _make_dict(keys, vals):
    return dict_(zip(keys, vals))
defvjp(_make_dict,
       lambda ans, keys, vals: lambda g:
       list(g[key] for key in keys), argnums=(1,))

class ContainerVSpace(VSpace):
    def __init__(self, value):
        self.shape = value
        self.shape = self._map(vspace)

    @property
    def size(self): return sum(self._values(self._map(lambda vs: vs.size)))
    def zeros(self): return self._map(lambda vs: vs.zeros())
    def ones(self):  return self._map(lambda vs: vs.ones())
    def randn(self): return self._map(lambda vs: vs.randn())
    def standard_basis(self):
        zero = self.zeros()
        for i, vs in self._kv_pairs(self.shape):
            for x in vs.standard_basis():
                yield self._subval(zero, i, x)
    def _add(self, xs, ys):
        return self._map(lambda vs, x, y: vs._add(x, y), xs, ys)
    def _mut_add(self, xs, ys):
        return self._map(lambda vs, x, y: vs._mut_add(x, y), xs, ys)
    def _scalar_mul(self, xs, a):
        return self._map(lambda vs, x: vs._scalar_mul(x, a), xs)
    def _inner_prod(self, xs, ys):
        return sum(self._values(self._map(lambda vs, x, y: vs._inner_prod(x, y), xs, ys)))
    def _covector(self, xs):
        return self._map(lambda vs, x: vs._covector(x), xs)

class SequenceVSpace(ContainerVSpace):
    def _values(self, x): return x
    def _kv_pairs(self, x): return enumerate(x)
    def _map(self, f, *args):
        return self.seq_type(map(f, self.shape, *args))
    def _subval(self, xs, idx, x):
        return self.seq_type(subvals(xs, [(idx, x)]))

class ListVSpace(SequenceVSpace):  seq_type = list_
class TupleVSpace(SequenceVSpace): seq_type = tuple_
class DictVSpace(ContainerVSpace):
    def _values(self, x):   return x.values()
    def _kv_pairs(self, x): return x.items()
    def _map(self, f, *args):return {k: f(vs, *[x[k] for x in args])
                                     for k, vs in self.shape.items()}
    def _subval(self, xs, idx, x):
        d = dict(xs.items())
        d[idx] = x
        return d

ListVSpace.register(list_)
TupleVSpace.register(tuple_)
DictVSpace.register(dict_)

class NamedTupleVSpace(SequenceVSpace):
    def _map(self, f, *args):
        return self.seq_type(*map(f, self.shape, *args))
    def _subval(self, xs, idx, x):
        return self.seq_type(*subvals(xs, [(idx, x)]))
