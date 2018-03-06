from itertools import count
from functools import reduce
from operator import attrgetter
from .tracer import trace, primitive, Node, Box, isbox, getval
from .fmap_util import fmap_to_zipped, fmap_to_list, container_fmap
from .util import func, subval, toposort

# -------------------- reverse mode --------------------

def make_vjp(fun, xs):
    fmap_in = fmap_out = container_fmap
    start_nodes = fmap_in(lambda _: VJPNode(None), xs)
    end_values, end_nodes, lfmap_out = trace(start_nodes, fun, xs, fmap_in, fmap_out)
    def vjp(g):
        return backward_pass(g, xs, start_nodes, end_nodes, fmap_in, lfmap_out)
    return vjp, end_values

def backward_pass(gs, xs, start_nodes, end_nodes, fmap_in, fmap_out):
    outgrads = {}
    final_fnode = VJPFunctionNode(lambda _: gs, (), lambda *a: (),
                                  end_nodes, fmap_out)
    for fnode in toposort(final_fnode, attrgetter('parent_fnodes')):
        parent_outgrads = fnode.vjp(fnode.children_fmap(lambda n: outgrads[n][0],
                                                        fnode.children))
        for p, p_outgrad in fmap_to_zipped(
                fnode.parent_fmap, fnode.parents, parent_outgrads):
            outgrads[p] = add_outgrads(outgrads.get(p), p_outgrad)

    return fmap_in(lambda n, x: (outgrads.get(n) or [vspace(x).zeros()])[0],
                   start_nodes, xs)

class VJPNode(Node):
    __slots__ = ['fun']
    def __init__(self, fun):
        self.fun = fun

    def process_primitive(self, ans, fun, args, kwargs, parents, parent_fmap):
        try:
            vjpmaker = primitive_vjps[fun]
        except KeyError:
            fun_name = getattr(fun, '__name__', fun)
            raise NotImplementedError("VJP of {} not defined".format(fun_name))

        vjp = vjpmaker(parent_fmap, ans, *args, **kwargs)
        fun_node = VJPFunctionNode(vjp, None, fun._fmap_out, parents, parent_fmap)
        output_nodes = fun._fmap_out(lambda _: VJPNode(fun_node), ans)
        fun_node.children = output_nodes
        return output_nodes, fun._fmap_out

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

def defvjp_argnum(fun, vjpmaker):
    assert fun._fmap is map
    def vjp_full(parent_fmap, ans, *args, **kwargs):
        vjps = parent_fmap(lambda argnum: vjpmaker(argnum, ans, args, kwargs),
                           range(len(args)))
        return lambda g: parent_fmap(lambda vjp: vjp(g), vjps)
    defvjp_full(fun, vjp_full)

def defvjp(fun, *vjpmakers):
    def vjp_full(parent_fmap, ans, *args, **kwargs):
        vjps = parent_fmap(lambda vjpmaker:
                           vjpmaker(ans, *args, **kwargs), vjpmakers)
        return lambda g: parent_fmap(lambda vjp: vjp(g), vjps)

    defvjp_full(fun, vjp_full)

# -------------------- forward mode --------------------

def make_jvp(fun, xs):
    fmap_in = fmap_out = container_fmap
    def jvp(gs):
        start_nodes = fmap_in(JVPNode, gs)
        end_values, end_nodes, _ = trace(start_nodes, fun, xs, fmap_in, fmap_out)
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
        gs = jvpmaker(parents, parent_gs, ans, *args, **kwargs)
        output_nodes = fun._fmap_out(JVPNode, gs)
        return output_nodes, fun._fmap_out

primitive_jvps = {}
def defjvp_full(fun, jvp_full):
    primitive_jvps[fun] = jvp_full

def defjvp_argnum(fun, jvpmaker):
    assert fun._fmap is map
    def jvp_full(parents, gs, ans, *args, **kwargs):
        return sum_outgrads(fun._fmap,
                            [jvpmaker(argnum, g, ans, args, kwargs)
                             for argnum, parent, g in zip(count(), parents, gs) if parent])
    defjvp_full(fun, jvp_full)

def defjvp(fun, *jvpfuns):
    def jvp_full(parents, gs, ans, *args, **kwargs):
        return sum_outgrads(
            fun._fmap,
            fun._fmap(
                lambda jvpfun, g: None if g is None else jvpfun(g, ans, *args, **kwargs),
                jvpfuns, gs))

    defjvp_full(fun, jvp_full)

def def_linear(fun):
    """Flags that a function is linear wrt all args"""
    defjvp_argnum(fun, lambda argnum, g, ans, args, kwargs:
                  fun(*subval(args, argnum, g), **kwargs))

def defjvp_is_fun(fun):
    def jvp_full(parents, gs, ans, *args, **kwargs):
        new_args = fun._fmap(lambda p, g, arg: arg if p is None else g,
                             parents, gs, args)
        return fun(*new_args, **kwargs)
    defjvp_full(fun, jvp_full)

# -------------------- vector behavior --------------------

def add_outgrads(prev_g_flagged, g):
    sparse = type(g) in sparse_object_types
    if prev_g_flagged:
        vs = vspace(g)
        prev_g, mutable = prev_g_flagged
        if mutable:
            if sparse:
                return sparse_add(vs, prev_g, g), True
            else:
                return vs.mut_add(prev_g, g), True
        else:
            if sparse:
                prev_g_mutable = vs.mut_add(None, prev_g)
                return sparse_add(vs, prev_g_mutable, g), True
            else:
                return vs.add(prev_g, g), True
    else:
        if sparse:
            return sparse_add(vspace(g), None, g), True
        else:
            return g, False

def sum_outgrads(fmap, gs):
    outgrads = []
    def accumulate(x):
        if x is not None:
            outgrads.append(x)
    fmap(accumulate, gs)
    return reduce(add_outgrads, outgrads, None)[0]

@primitive
def sparse_add(vs, x_prev, x_new):
    x_prev = x_prev if x_prev is not None else vs.zeros()
    return x_new.mut_add(x_prev)

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
    def mut_add(self, x_prev, x_new):
      x_prev = x_prev if x_prev is not None else self.zeros()
      return self._mut_add(x_prev, x_new)
    @primitive
    def add(self, x_prev, x_new):     return self._add(x_prev, x_new)
    @primitive
    def scalar_mul(self, x, a):       return self._scalar_mul(x, a)
    @primitive
    def inner_prod(self, x, y):       return self._inner_prod(x, y)
    @primitive
    def covector(self, x):            return self._covector(x)

    def _add(self, x, y):        return x + y
    def _mut_add(self, x, y):    x += y; return x
    def _scalar_mul(self, x, a): return x * a
    def _inner_prod(self, x, y): assert False
    def _covector(self, x):      return x

    def __eq__(self, other):
        return type(self) == type(other) and self.__dict__ == other.__dict__

    def __repr__(self):
        return "{}_{}".format(type(self).__name__, self.__dict__)

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

class SparseBox(Box):
    __slots__ = []
class SparseObject(object):
    __slots__ = ['vs', 'mut_add']
    def __init__(self, vs, mut_add):
        self.vs = vs
        self.mut_add = mut_add
VSpace.register(SparseObject, lambda x : x.vs)
SparseBox.register(SparseObject)
sparse_object_types = {SparseObject, SparseBox}

# -------------------- core reverse mode grads --------------------

identity_vjp = lambda argnums, *args: lambda g: g
defvjp(sparse_add, None, identity_vjp, identity_vjp)
defvjp(func(VSpace.add    ), None, identity_vjp, identity_vjp)
defvjp(func(VSpace.mut_add), None, identity_vjp, identity_vjp)
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
defjvp(sparse_add, None, identity_jvp, identity_jvp)
defjvp(func(VSpace.mut_add), None, identity_jvp, identity_jvp)
defjvp(func(VSpace.add),     None, identity_jvp, identity_jvp)
def_linear(func(VSpace.scalar_mul))
def_linear(func(VSpace.inner_prod))
defjvp_is_fun(func(VSpace.covector))
