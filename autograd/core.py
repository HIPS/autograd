from itertools import count
from functools import reduce, partial
from .tracer import trace, primitive, Node, Box, isbox, getval
from .fmap_util import fmap_to_list, container_fmap, select_map, bound_fmap
from .util import func, subval, toposort

# -------------------- reverse mode --------------------

def make_vjp(fun, xs):
    fmap_in = fmap_out = container_fmap
    start_nodes = fmap_in(partial(VJPNode, None), xs)
    end_values, end_nodes = trace((start_nodes,), fun, (xs,), fmap_in, fmap_out)
    start_fmap = bound_fmap(fmap_in , start_nodes)
    end_fmap   = bound_fmap(fmap_out, end_nodes)
    def vjp(g):
        return backward_pass(start_fmap, end_fmap, g)
    return vjp, end_values

def backward_pass(start_fmap, end_fmap, end_gs):
    gs = {}
    get_g     = lambda n   : n.vspace.densify(gs.get(n))
    update_gs = lambda n, g: gs.update({n: n.vspace.add(gs.get(n), g)})
    end_fmap(update_gs, end_gs)
    for fnode in toposort(get_fnodes(end_fmap)):
        child_gs = fnode.child_fmap(get_g)
        parent_gs = fnode.vjp(child_gs)
        fnode.parent_fmap(update_gs, parent_gs)
    return start_fmap(get_g)

class VJPNode(Node):
    __slots__ = ['fun', 'vspace']
    def __init__(self, fun, x):
        self.fun = fun
        self.vspace = vspace(x)

    def process_primitive(self, ans, fun, args, kwargs, parents):
        try:
            vjpmaker = primitive_vjps[fun]
        except KeyError:
            fun_name = getattr(fun, '__name__', fun)
            raise NotImplementedError("VJP of {} not defined".format(fun_name))

        if vjpmaker is None:
            return fun.fmap_out(lambda _: None, ans)
        vjp = vjpmaker(parents, ans, *args, **kwargs)
        fnode = VJPFunction()
        children = fun.fmap_out(partial(VJPNode, fnode), ans)
        parent_fmap = bound_fmap(fun.fmap_in, parents)
        child_fmap  = bound_fmap(fun.fmap_out, children)
        fnode.initialize(vjp, parent_fmap, child_fmap)
        return children

class VJPFunction(object):
    __slots__ = ['vjp', 'child_fmap', 'parent_fmap', 'parent_fnodes']
    def __init__(self): pass
    def initialize(self, vjp, parent_fmap, child_fmap):
        self.vjp = vjp
        self.child_fmap = child_fmap
        self.parent_fmap = parent_fmap
        self.parent_fnodes = get_fnodes(parent_fmap)

def get_fnodes(fmap):
    fnodes = set()
    fmap(lambda p: p.fun and fnodes.add(p.fun))
    return fnodes

primitive_vjps = {}
def defvjp_full(fun, vjpmaker):
    primitive_vjps[fun] = vjpmaker

def defvjp(fun, *vjpmakers):
    def vjp_full(parents, ans, *args, **kwargs):
        vjps = fun.fmap_in(
            lambda p, vjpmaker:
            p and vjpmaker(ans, *args, **kwargs), parents, vjpmakers)
        return lambda g: fun.fmap_in(lambda p, vjp: p and vjp(g), parents, vjps)
    defvjp_full(fun, vjp_full)

def defvjp_zero(fun):
    defvjp_full(fun, None)

# -------------------- forward mode --------------------

def make_jvp(fun, xs):
    fmap_in = fmap_out = container_fmap
    def jvp(gs):
        start_nodes = fmap_in(JVPNode, gs)
        end_values, end_nodes = trace((start_nodes,), fun, (xs,),
                                      fmap_in, fmap_out)
        gs_out = fmap_out(lambda n, v: vspace(v).zeros() if n is None else n.g,
                          end_nodes, end_values)
        return end_values, gs_out
    return jvp

class JVPNode(Node):
    __slots__ = ['g']
    def __init__(self, g):
        self.g = g

    def process_primitive(self, ans, fun, args, kwargs, parents):
        parent_gs = fun.fmap_in(lambda p: p and p.g, parents)
        try:
            jvp_full = primitive_jvps[fun]
        except KeyError:
            name = getattr(fun, '__name__', fun)
            raise NotImplementedError("JVP of {}".format(name))
        child_gs = jvp_full(parent_gs, ans, *args, **kwargs)
        output_nodes = fun.fmap_out(
            lambda g: None if g is None else JVPNode(g), child_gs)
        return output_nodes

primitive_jvps = {}
def defjvp_full(fun, jvp_full):
    primitive_jvps[fun] = jvp_full

def defjvp(fun, *jvpfuns):
    def jvp_full(gs, ans, *args, **kwargs):
        vs = vspace(ans)
        return vs.sum(
            fmap_to_list(fun.fmap_in, fun.fmap_in(
                lambda jvpfun, g: None if g is None
                else jvpfun(g, ans, *args, **kwargs),
                jvpfuns, gs)))

    defjvp_full(fun, jvp_full)

def def_multilinear(fun):
    """Flags that a function is linear wrt all args"""
    assert fun.fmap_in is map
    def jvp_full(gs, ans, *args, **kwargs):
        vs = vspace(ans)
        return vs.sum([fun(*subval(args, argnum, g), **kwargs)
                       for argnum, g in zip(count(), gs) if g is not None])
    defjvp_full(fun, jvp_full)

def defjvp_is_fun(fun, fmap=select_map([0])):
    def jvp_full(gs, ans, *args, **kwargs):
        sub_val = lambda arg, g: vspace(arg).zeros() if g is None else g
        new_args = fmap(sub_val, args, gs)
        return fun(*new_args, **kwargs)
    defjvp_full(fun, jvp_full)

def defjvp_zero(fun):
    defjvp_full(fun, lambda parent_gs, ans, *args, **kwargs:
                fun.fmap_out(lambda _: None, ans))

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
