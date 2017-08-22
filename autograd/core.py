from collections import defaultdict
from functools import partial
from .tracer import (trace, primitive, notrace_primitive, Node, Box,
                     register_box, toposort)
from .vspace import vspace, assert_vspace_match, register_vspace, VSpace
from .util import unary_to_nary, func, subval

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
    assert_vspace_match(outgrads[end_node][0], end_node.vspace)
    for node in toposort(end_node):
        cur_outgrad = outgrads.pop(node)
        for parent, vjp in node.parents_and_vjps:
            outgrad = vjp(cur_outgrad[0])
            assert_vspace_match(outgrad, parent.vspace)
            outgrads[parent] = add_outgrads(parent.vspace, outgrads.get(parent), outgrad)
    return cur_outgrad[0]

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
    __slots__ = ['vspace', 'parents', 'parents_and_vjps']
    def __init__(self, value, fun, args, kwargs, parent_argnums, parents):
        self.vspace = vspace(value)
        self.parents = parents
        self.parents_and_vjps = [
            (parent, primitive_vjp(fun, argnum, value, parent.vspace,
                                   self.vspace, args, kwargs))
            for argnum, parent in zip(parent_argnums, parents)]

    def initialize_root(self, value):
        self.vspace = vspace(value)
        self.parents = []
        self.parents_and_vjps = []

class JVPNode(Node):
    __slots__ = ['vspace', 'g']
    def __init__(self, value, fun, args, kwargs, parent_argnums, parents):
        self.vspace = vspace(value)
        cur_g = None
        for argnum, parent in zip(parent_argnums, parents):
            new_g = primitive_jvp(fun, argnum, parent.g, value, vspace(parent.g), self.vspace, args, kwargs)
            assert_vspace_match(new_g, self.vspace)
            cur_g = add_outgrads(parent.vspace, cur_g, new_g)

        self.g = cur_g[0]

    def initialize_root(self, x, g):
        self.vspace = vspace(x)
        assert_vspace_match(g, self.vspace)
        self.g = g

def add_outgrads(vs, prev_g_flagged, g):
    sparse = type(g) in sparse_object_types
    if prev_g_flagged:
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
            return sparse_add(vs.zeros(), g), True
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

def zero_vjp(ans, vs, gvs, *args, **kwargs):
    return lambda g: vs.zeros()

primitive_vjps = defaultdict(dict)

def primitive_vjp(fun, argnum, ans, vs, gvs, args, kwargs):
    try:
        vjp = primitive_vjps[fun][argnum]
    except KeyError:
        if primitive_vjps[fun]:
            errstr = "Gradient of {0} w.r.t. arg number {1} not yet implemented."
        else:
            errstr = "Gradient of {0} not yet implemented."
        raise NotImplementedError(errstr.format(repr(fun), argnum))
    return vjp(ans, vs, gvs, args, kwargs)

def defvjp(fun, vjpmaker, argnum=0):
    def vjp_fixed_args(ans, vs, gvs, args, kwargs):
        return vjpmaker(ans, vs, gvs, *args, **kwargs)
    primitive_vjps[fun][argnum] = vjp_fixed_args

def defvjps(fun, vjpmaker, argnums):
    for argnum in argnums:
        defvjp(fun, partial(vjpmaker, argnum), argnum)

def defvjp_argnum(fun, vjpmaker):
    primitive_vjps[fun] = first_arg_as_get(vjpmaker)

def defvjp_is_zero(fun, argnums=(0,)):
    for argnum in argnums:
        defvjp(fun, zero_vjp, argnum)
        defjvp(fun, zero_jvp, argnum)

class first_arg_as_get(object):
    def __init__(self, f):
        self.f = f
    def __getitem__(self, argnum):
        return lambda *args, **kwargs: self.f(argnum, *args, **kwargs)

identity_vjp = lambda *args: lambda g: g

@primitive
def sparse_add(x_prev, x_new): return x_new.mut_add(x_prev)
defvjps(sparse_add, identity_vjp, argnums=[0, 1])

defvjps(func(VSpace.mut_add), identity_vjp, argnums=[1,2])
defvjp(func(VSpace.inner_prod), lambda ans, vs, gvs, vs_, x, y: lambda g:
       vs.covector(vs.scalar_mul(y, gvs.covector(g))), argnum=1)
defvjp(func(VSpace.inner_prod), lambda ans, vs, gvs, vs_, x, y: lambda g:
       vs.covector(vs.scalar_mul(x, gvs.covector(g))), argnum=2)
defvjps(func(VSpace.add), identity_vjp, argnums=[1,2])
defvjp(func(VSpace.covector), lambda ans, vs, gvs, vs_, x: lambda g:
       gvs.covector(g), argnum=1)
defvjp(func(VSpace.scalar_mul), lambda ans, vs, gvs, vs_, x, a: lambda g:
       vs.covector(gvs.scalar_mul(gvs.covector(g), a)), argnum=1)
defvjp(func(VSpace.scalar_mul), lambda ans, vs, gvs, vs_, x, a: lambda g:
       gvs.inner_prod(g, gvs.covector(x)), argnum=2)

def primitive_jvp(fun, argnum, g, ans, gvs, vs, args, kwargs):
    try:
        return primitive_jvps[fun][argnum](g, ans, gvs, vs, *args, **kwargs)
    except KeyError:
        raise NotImplementedError("JVP of {} wrt arg number {} not yet implemented"
                                  .format(fun.__name__, argnum))

primitive_jvps = defaultdict(dict)

def zero_jvp(g, ans, *args, **kwargs): return vspace(ans).zeros()

def defjvp(fun, jvpfun, argnum=0):
    primitive_jvps[fun][argnum] = jvpfun

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
    defjvp(fun, lambda g, ans, gvs, vs, *args, **kwargs:
           fun(*subval(args, argnum, g), **kwargs), argnum=argnum)

def def_linear_wrt_args(fun, argnums):
    for argnum in argnums:
        def_linear_wrt_arg(fun, argnum)

def def_multilinear(fun):
    """
    This is to flag that a function is linear in all of its args.
    """
    defjvp_argnum(fun, lambda argnum, g, ans, gvs, vs, *args, **kwargs:
                  fun(*subval(args, argnum, g), **kwargs))
