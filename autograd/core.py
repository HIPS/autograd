from itertools import count
from functools import reduce, partial
from operator import attrgetter
from .tracer import trace, primitive, Node, Box, isbox, getval
from .fmap_util import fmap_to_zipped, fmap_to_list, container_fmap, select_map
from .util import func, subval, toposort

# -------------------- reverse mode --------------------

def make_vjp(fun, xs):
    fmap_in = fmap_out = container_fmap
    start_nodes = fmap_in(partial(VJPNode, None), xs)
    end_values, end_nodes, lfmap_out = trace((start_nodes,), fun, (xs,), fmap_in, fmap_out)
    def vjp(g):
        return backward_pass(g, xs, start_nodes, end_nodes, fmap_in, lfmap_out)
    return vjp, end_values

def backward_pass(gs, xs, start_nodes, end_nodes, fmap_in, fmap_out):
    outgrads = {}
    final_fnode = VJPFunctionNode(lambda _: gs, (), lambda *a: (),
                                  end_nodes, fmap_out)
    for fnode in toposort(final_fnode, attrgetter('parent_fnodes')):
        children_gs = fnode.children_fmap(
            lambda n: n.vspace.densify(outgrads.get(n)), fnode.children)
        parent_gs = fnode.vjp(children_gs)
        for p, g in fmap_to_zipped(fnode.parent_fmap, fnode.parents, parent_gs):
            outgrads[p] = p.vspace.add(outgrads.get(p), g)

    return fmap_in(lambda n: n.vspace.densify(outgrads.get(n)), start_nodes)

class VJPNode(Node):
    __slots__ = ['fun', 'vspace']
    def __init__(self, fun, x):
        self.fun = fun
        self.vspace = vspace(x)

    def process_primitive(self, ans, fun, args, kwargs, parents, parent_fmap):
        try:
            vjpmaker = primitive_vjps[fun]
        except KeyError:
            fun_name = getattr(fun, '__name__', fun)
            raise NotImplementedError("VJP of {} not defined".format(fun_name))

        if vjpmaker is None:
            return fun._fmap_out(lambda _: None, ans)
        vjp = vjpmaker(parent_fmap, ans, *args, **kwargs)
        fun_node = VJPFunctionNode(vjp, None, fun._fmap_out, parents, parent_fmap)
        output_nodes = fun._fmap_out(partial(VJPNode, fun_node), ans)
        fun_node.children = output_nodes
        return output_nodes

class VJPFunctionNode(object):
    def __init__(self, vjp, children, children_fmap, parents, parent_fmap):
        self.vjp = vjp
        self.children = children
        self.children_fmap = children_fmap
        self.parents = parents
        self.parent_fmap = parent_fmap
        self.parent_fnodes = set(filter(bool, fmap_to_list(
            parent_fmap, parent_fmap(attrgetter('fun'), parents))))

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
            jvpmaker = primitive_jvps[fun]
        except KeyError:
            name = getattr(fun, '__name__', fun)
            raise NotImplementedError("JVP of {}".format(name))
        if jvpmaker is None:
            return fun._fmap_out(lambda _: None, ans)
        gs = jvpmaker(parents, parent_gs, ans, *args, **kwargs)
        output_nodes = fun._fmap_out(lambda g: None if g is None else JVPNode(g), gs)
        return output_nodes

primitive_jvps = {}
def defjvp_full(fun, jvp_full):
    primitive_jvps[fun] = jvp_full

def defjvp(fun, *jvpfuns):
    def jvp_full(parents, gs, ans, *args, **kwargs):
        vs = vspace(ans)
        return vs.sum(
            fmap_to_list(fun._fmap,
            fun._fmap(
                lambda jvpfun, g: None if g is None else jvpfun(g, ans, *args, **kwargs),
                jvpfuns, gs)))

    defjvp_full(fun, jvp_full)

def def_linear(fun):
    """Flags that a function is linear wrt all args"""
    assert fun._fmap is map
    def jvp_full(parents, gs, ans, *args, **kwargs):
        vs = vspace(ans)
        return vs.sum([fun(*subval(args, argnum, g), **kwargs)
                       for argnum, parent, g in zip(count(), parents, gs) if parent])
    defjvp_full(fun, jvp_full)

def defjvp_is_fun(fun, parent_fmap=select_map([0])):
    def jvp_full(parents, gs, ans, *args, **kwargs):
        sub_val = lambda arg, g: vspace(arg).zeros() if g is None else g
        new_args = parent_fmap(sub_val, args, gs)
        return fun(*new_args, **kwargs)
    defjvp_full(fun, jvp_full)

def defjvp_zero(fun):
    defjvp_full(fun, lambda parents, parent_gs, ans, *args, **kwargs:
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
def_linear(func(VSpace.scalar_mul))
def_linear(func(VSpace.inner_prod))
defjvp_is_fun(func(VSpace.covector), select_map([1]))
