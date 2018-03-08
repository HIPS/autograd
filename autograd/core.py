from itertools import count
from functools import reduce, partial
from collections import namedtuple
from operator import attrgetter
from .tracer import trace, primitive, Node, Box, isbox, getval
from .fmap_util import fmap_to_zipped, fmap_to_list, container_fmap, select_map
from .util import func, subval, toposort

# -------------------- reverse mode --------------------

def make_vjp(fun, xs):
    fmap_in = fmap_out = container_fmap
    start_fnode, children = source_fnode(xs, fmap_in)
    start_nodes = fmap_in(partial(VJPNode, start_fnode), xs, children)
    end_values, end_nodes, lfmap_out = trace((start_nodes,), fun, (xs,), fmap_in, fmap_out)
    def vjp(g):
        final_fnode = sink_fnode(g, end_nodes, lfmap_out, start_fnode)
        return backward_pass(final_fnode)
    return vjp, end_values

def backward_pass(final_fnode):
    outgrads = {}
    for fnode in toposort(final_fnode):
        child_gs = fnode.child_fmap(lambda n: n.vspace.densify(outgrads.get(n)),
                                    fnode.children)
        parent_gs = fnode.vjp(child_gs)
        for p, g in fmap_to_zipped(fnode.parent_fmap, fnode.parents, parent_gs):
            outgrads[p] = p.vspace.add(outgrads.get(p), g)

    return child_gs

class VJPNode(Node):
    __slots__ = ['fun', 'var']
    def __init__(self, fun, x, var):
        self.fun = fun
        self.var = var

    def process_primitive(self, ans, fun, args, kwargs, parents, parent_fmap):
        try:
            vjpmaker = primitive_vjps[fun]
        except KeyError:
            fun_name = getattr(fun, '__name__', fun)
            raise NotImplementedError("VJP of {} not defined".format(fun_name))

        if vjpmaker is None:
            return fun._fmap_out(lambda _: None, ans)
        vjp = vjpmaker(parent_fmap, ans, *args, **kwargs)
        fun_node, children = new_fnode(vjp, ans, fun._fmap_out, parents, parent_fmap)
        output_nodes = fun._fmap_out(partial(VJPNode, fun_node), ans, children)
        return output_nodes

def source_fnode(xs, fmap_in):
    return new_fnode(lambda x: (), xs, fmap_in, (), map)

def sink_fnode(g, nodes, fmap_out, start_fnode):
    fnode, _ = new_fnode(lambda _: g, (), map, nodes, fmap_out)
    fnode.parent_fnodes.add(start_fnode)
    return fnode

def new_fnode(vjp, xs, child_fmap, parent_nodes, parent_fmap):
    children = child_fmap(lambda x: VJPVariable(vspace(x)), xs)
    parents = parent_fmap(attrgetter('var'), parent_nodes)
    parent_fnodes = set(fmap_to_list(parent_fmap, parent_fmap(attrgetter('fun'), parent_nodes)))
    return VJPFunction(vjp, children, child_fmap, parents, parent_fmap, parent_fnodes), children

class VJPFunction(object):
    def __init__(self, vjp, children, child_fmap, parents, parent_fmap, parent_fnodes):
        self.vjp = vjp
        self.children = children
        self.child_fmap = child_fmap
        self.parents = parents
        self.parent_fmap = parent_fmap
        self.parent_fnodes = parent_fnodes

class VJPVariable(object):
    def __init__(self, vspace):
        self.vspace = vspace

primitive_vjps = {}
def defvjp_full(fun, vjpmaker):
    primitive_vjps[fun] = vjpmaker

def defvjp(fun, *vjpmakers):
    def vjp_full(parent_fmap, ans, *args, **kwargs):
        vjps = parent_fmap(lambda vjpmaker:
                           vjpmaker(ans, *args, **kwargs), vjpmakers)
        return lambda g: parent_fmap(lambda vjp: vjp(g), vjps)

    defvjp_full(fun, vjp_full)

def defvjp_zero(fun):
    defvjp_full(fun, None)

# -------------------- forward mode --------------------

def make_jvp(fun, xs):
    fmap_in = fmap_out = container_fmap
    def jvp(gs):
        start_nodes = fmap_in(JVPNode, gs)
        end_values, end_nodes, _ = trace((start_nodes,), fun, (xs,), fmap_in, fmap_out)
        gs_out = fmap_out(lambda n, v: vspace(v).zeros() if n is None else n.g,
                          end_nodes, end_values)
        return end_values, gs_out
    return jvp

class JVPNode(Node):
    __slots__ = ['g']
    def __init__(self, g):
        self.g = g

    def process_primitive(self, ans, fun, args, kwargs, parents, parent_fmap):
        parent_gs = parent_fmap(attrgetter('g'), parents)
        try:
            jvp_full = primitive_jvps[fun]
        except KeyError:
            name = getattr(fun, '__name__', fun)
            raise NotImplementedError("JVP of {}".format(name))
        child_gs = jvp_full(parent_gs, ans, *args, **kwargs)
        output_nodes = fun._fmap_out(
            lambda g: None if g is None else JVPNode(g), child_gs)
        return output_nodes

primitive_jvps = {}
def defjvp_full(fun, jvp_full):
    primitive_jvps[fun] = jvp_full

def defjvp(fun, *jvpfuns):
    def jvp_full(gs, ans, *args, **kwargs):
        vs = vspace(ans)
        return vs.sum(
            fmap_to_list(fun._fmap, fun._fmap(
                lambda jvpfun, g: None if g is None
                else jvpfun(g, ans, *args, **kwargs),
                jvpfuns, gs)))

    defjvp_full(fun, jvp_full)

def def_multilinear(fun):
    """Flags that a function is linear wrt all args"""
    assert fun._fmap is map
    def jvp_full(gs, ans, *args, **kwargs):
        vs = vspace(ans)
        return vs.sum([fun(*subval(args, argnum, g), **kwargs)
                       for argnum, g in zip(count(), gs) if g is not None])
    defjvp_full(fun, jvp_full)

def defjvp_is_fun(fun, parent_fmap=select_map([0])):
    def jvp_full(gs, ans, *args, **kwargs):
        sub_val = lambda arg, g: vspace(arg).zeros() if g is None else g
        new_args = parent_fmap(sub_val, args, gs)
        return fun(*new_args, **kwargs)
    defjvp_full(fun, jvp_full)

def defjvp_zero(fun):
    defjvp_full(fun, lambda parent_gs, ans, *args, **kwargs:
                fun._fmap_out(lambda _: None, ans))

# -------------------- vector behavior --------------------

class VSpace(object):
    __slots__ = []
    mappings = {}
    iscomplex = False
    def __init__(self, value): pass

    def zeros(self):          assert False, repr(self)
    def ones(self):           assert False, repr(self)
    def standard_basis(self): assert False, repr(self)
    def randn(self):          assert False, repr(self)

    @primitive
    def add_not_none(self, x, y):     return self._add(x, y)
    @primitive
    def scalar_mul(self, x, a):       return self._scalar_mul(x, a)
    @primitive
    def inner_prod(self, x, y):       return self._inner_prod(x, y)
    @primitive
    def covector(self, x):            return self._covector(x)

    def _add(self, x, y):        return x + y
    def _scalar_mul(self, x, a): return x * a
    def _inner_prod(self, x, y): assert False
    def _covector(self, x):      return x

    def __eq__(self, other):
        return type(self) == type(other) and self.__dict__ == other.__dict__

    def __repr__(self):
        return "{}_{}".format(type(self).__name__, self.__dict__)

    def densify(self, x):
        if x is None:
            return self.zeros()
        else:
            return x

    def add(self, prev, new):
        return new if prev is None else self.add_not_none(prev, new)

    def sum(self, xs):
        xs = [x for x in xs if x is not None]
        xs_sum = reduce(self.add_not_none, xs)
        return self.densify(xs_sum)

    @classmethod
    def register(cls, value_type, vspace_maker=None):
        if vspace_maker:
            VSpace.mappings[value_type] = vspace_maker
        else:
            VSpace.mappings[value_type] = cls

def vspace(value):
    try:
        return VSpace.mappings[type(value)](value)
    except KeyError:
        if isbox(value):
            return vspace(getval(value))
        else:
            raise TypeError("Can't find vector space for value {} of type {}. "
                            "Valid types are {}".format(
                                value, type(value), VSpace.mappings.keys()))

# -------------------- core reverse mode grads --------------------

identity_vjp = lambda argnums, *args: lambda g: g
defvjp(func(VSpace.add_not_none), None, identity_vjp, identity_vjp)
defvjp(func(VSpace.inner_prod), None,
       lambda ans, vs, x, y: lambda g:  vs.covector(vs.scalar_mul(y, g)),
       lambda ans, vs, x, y: lambda g:  vs.covector(vs.scalar_mul(x, g)))
defvjp(func(VSpace.covector), None,
          lambda ans, vs, x: lambda g: vs.covector(g))
defvjp(func(VSpace.scalar_mul), None,
       lambda ans, vs, x, a: lambda g: vs.covector(vs.scalar_mul(vs.covector(g), a)),
       lambda ans, vs, x, a: lambda g: vs.inner_prod(g, vs.covector(x)))

# -------------------- core forward mode grads --------------------

identity_jvp = lambda g, *args, **kwargs: g
defjvp(func(VSpace.add_not_none), None, identity_jvp, identity_jvp)
def_multilinear(func(VSpace.scalar_mul))
def_multilinear(func(VSpace.inner_prod))
defjvp_is_fun(func(VSpace.covector), select_map([1]))
