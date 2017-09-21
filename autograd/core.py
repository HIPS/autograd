from collections import defaultdict
from functools import partial
from .tracer import (trace, primitive, notrace_primitive, Node, Box,
                     register_box, toposort)
from .vspace import vspace, register_vspace, VSpace
from .util import func, subval

def make_vjp(fun, x):
    start_node = VJPNode.new_root(x)
    end_value, end_node =  trace(start_node, fun, x)
    if end_node is None:
        def vjp(g): return vspace(x).zeros()
    else:
        def vjp(g): return backward_pass(g, end_node)
    return vjp, end_value

def backward_pass(g, end_node):
    outgrads = {end_node : (g, False)}
    for node in toposort(end_node):
        outgrad = outgrads.pop(node)
        ingrads = node.vjp(outgrad[0])
        for parent, ingrad in zip(node.parents, ingrads):
            outgrads[parent] = add_outgrads(outgrads.get(parent), ingrad)
    return outgrad[0]

def make_jvp(fun, x):
    def jvp(g):
        start_node = JVPNode.new_root(x, g)
        end_value, end_node = trace(start_node, fun, x)
        if end_node is None:
            return vspace(end_value).zeros()
        else:
            return end_node.g
    return jvp

class VJPNode(Node):
    __slots__ = ['parents', 'vjp']
    def __init__(self, value, fun, args, kwargs, parent_argnums, parents):
        self.parents = parents
        self.vjp = primitive_vjps[fun](parent_argnums, value, args, kwargs)

    def initialize_root(self, value):
        self.parents = []
        self.vjp = lambda g: ()

class JVPNode(Node):
    __slots__ = ['g']
    def __init__(self, value, fun, args, kwargs, parent_argnums, parents):
        cur_g = None
        for argnum, parent in zip(parent_argnums, parents):
            new_g = primitive_jvp(fun, argnum, parent.g, value, args, kwargs)
            cur_g = add_outgrads(cur_g, new_g)

        self.g = cur_g[0]

    def initialize_root(self, x, g):
        self.g = g

def add_outgrads(prev_g_flagged, g):
    sparse = type(g) in sparse_object_types
    if prev_g_flagged:
        vs = vspace(g)
        prev_g, mutable = prev_g_flagged
        if mutable:
            if sparse:
                return sparse_add(prev_g, g), True
            else:
                return vs.mut_add(prev_g, g), True
        else:
            if sparse:
                prev_g_mutable = vs.mut_add(vs.zeros(), prev_g)
                return sparse_add(prev_g_mutable, g), True
            else:
                return vs.add(prev_g, g), True
    else:
        if sparse:
            return sparse_add(vspace(g).zeros(), g), True
        else:
            return g, False

class SparseBox(Box):
    __slots__ = []
class SparseObject(object):
    __slots__ = ['vs', 'mut_add']
    def __init__(self, vs, mut_add):
        self.vs = vs
        self.mut_add = mut_add
register_vspace(lambda x : x.vs, SparseObject)
register_box(SparseBox, SparseObject)
sparse_object_types = set((SparseObject, SparseBox))

def zero_vjp(argnum):
    return lambda ans, *args, **kwargs: lambda g: vspace(args[argnum]).zeros()

primitive_vjps_onearg = defaultdict(dict)
primitive_vjps = {}

def defvjp_argnums(fun, vjpmaker):
    primitive_vjps[fun] = vjpmaker

def defvjp(fun, vjpmaker, argnum=0):
    primitive_vjps_onearg[fun][argnum] = vjpmaker
    vjps_dict = primitive_vjps_onearg[fun]
    def vjp_argnums(argnums, ans, args, kwargs):
        L = len(argnums)
        # These first two cases are purely for optimizations
        if L == 1:
            argnum = argnums[0]
            vjp = vjps_dict[argnum](ans, *args, **kwargs)
            return lambda g: (vjp(g),)
        elif L == 2:
            argnum_0, argnum_1 = argnums
            vjp_0 = vjps_dict[argnum_0](ans, *args, **kwargs)
            vjp_1 = vjps_dict[argnum_1](ans, *args, **kwargs)
            return lambda g: (vjp_0(g), vjp_1(g))
        else:
            vjps = [vjps_dict[argnum](ans, *args, **kwargs) for argnum in argnums]
            return lambda g: (vjp(g) for vjp in vjps)

    primitive_vjps[fun] = vjp_argnums

def defvjp_argnum(fun, vjpmaker):
    def vjp_argnums(argnums, *args):
        vjps = [vjpmaker(argnum, *args) for argnum in argnums]
        return lambda g: (vjp(g) for vjp in vjps)
    primitive_vjps[fun] = vjp_argnums

def defvjps(fun, vjpmaker, argnums):
    for argnum in argnums:
        defvjp(fun, partial(vjpmaker, argnum), argnum)

def defvjp_is_zero(fun, argnums=(0,)):
    for argnum in argnums:
        defvjp(fun, zero_vjp(argnum), argnum)
        defjvp(fun, zero_jvp, argnum)

identity_vjp = lambda *args: lambda g: g
identity_jvp = lambda argnum, g, *args, **kwargs: g

@primitive
def sparse_add(x_prev, x_new): return x_new.mut_add(x_prev)
defvjps(sparse_add, identity_vjp, argnums=[0, 1])

defvjps(func(VSpace.mut_add), identity_vjp, argnums=[1,2])
defvjp(func(VSpace.inner_prod), lambda ans, vs_, x, y: lambda g:
       vspace(x).covector(vspace(x).scalar_mul(y, vspace(g).covector(g))), argnum=1)
defvjp(func(VSpace.inner_prod), lambda ans, vs_, x, y: lambda g:
       vspace(x).covector(vspace(x).scalar_mul(x, vspace(g).covector(g))), argnum=2)
defvjps(func(VSpace.add), identity_vjp, argnums=[1,2])
defvjp(func(VSpace.covector), lambda ans, vs_, x: lambda g:
       vspace(g).covector(g), argnum=1)
defvjp(func(VSpace.scalar_mul), lambda ans, vs_, x, a: lambda g:
       vspace(x).covector(vspace(g).scalar_mul(vspace(g).covector(g), a)), argnum=1)
defvjp(func(VSpace.scalar_mul), lambda ans, vs_, x, a: lambda g:
       vspace(g).inner_prod(g, vspace(g).covector(x)), argnum=2)

class first_arg_as_get(object):
    def __init__(self, f):
        self.f = f
    def __getitem__(self, argnum):
        return lambda *args, **kwargs: self.f(argnum, *args, **kwargs)

def primitive_jvp(fun, argnum, g, ans, args, kwargs):
    try:
        return primitive_jvps[fun][argnum](g, ans, args, kwargs)
    except KeyError:
        raise NotImplementedError("JVP of {} wrt arg number {} not yet implemented"
                                  .format(fun.__name__, argnum))

primitive_jvps = defaultdict(dict)

def zero_jvp(g, ans, *args, **kwargs): return vspace(ans).zeros()

def defjvp(fun, jvpfun, argnum=0):
    def jvpfun_fixed_args(g, ans, args, kwargs):
        return jvpfun(g, ans, *args, **kwargs)
    primitive_jvps[fun][argnum] = jvpfun_fixed_args

def defjvps(fun, jvpfun, argnums):
    for argnum in argnums:
        defjvp(fun, partial(jvpfun, argnum), argnum)

def defjvp_argnum(fun, jvpmaker):
    primitive_jvps[fun] = first_arg_as_get(jvpmaker)

def def_linear_wrt_arg(fun, argnum=0):
    """
    This signifies that a function is linear in the sense of linear
    algebra/functional analysis:

    fun(a*x + b*y) = a*fun(x) + b*fun(y)

    In this case the jvp of fun is the same as fun itself.
    """
    defjvp(fun, lambda g, ans, *args, **kwargs:
           fun(*subval(args, argnum, g), **kwargs), argnum=argnum)

def def_linear_wrt_args(fun, argnums):
    for argnum in argnums:
        def_linear_wrt_arg(fun, argnum)

def def_multilinear(fun):
    """
    This is to flag that a function is linear in all of its args.
    """
    defjvp_argnum(fun, lambda argnum, g, ans, args, kwargs:
                  fun(*subval(args, argnum, g), **kwargs))

defjvps(sparse_add, identity_jvp, argnums=[0, 1])

defjvps(func(VSpace.mut_add), identity_jvp, argnums=[1,2])
def_multilinear(func(VSpace.inner_prod))
defjvps(func(VSpace.add), identity_jvp, argnums=[1,2])
defjvp(func(VSpace.covector), lambda g, ans, gvs_, x:
       vspace(x).covector(g), argnum=1)
def_multilinear(func(VSpace.scalar_mul))
