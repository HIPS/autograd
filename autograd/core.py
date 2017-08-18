from collections import defaultdict
from functools import partial
from .tracer import (trace, primitive, notrace_primitive, Node, Box,
                     register_box, toposort, getval)
from .vspace import vspace, assert_vspace_match, register_vspace, VSpace
from .util import unary_to_nary, func

def make_vjp(fun, x):
    end_value, end_node =  trace(VJPNode, fun, x)
    if end_node is None:
        def vjp(g): return vspace(x).zeros()
    else:
        def vjp(g): return backward_pass(g, end_node)
    return vjp, end_value

def backward_pass(g, end_node):
    outgrads = {end_node : (g, False)}
    assert_vspace_match(outgrads[end_node][0], end_node.vspace)
    for node in toposort(end_node):
        if node not in outgrads: continue
        cur_outgrad = outgrads.pop(node)
        for parent, vjp in zip(node.parents, node.vjps):
            outgrad = vjp(cur_outgrad[0])
            assert_vspace_match(outgrad, parent.vspace)
            outgrads[parent] = add_outgrads(parent.vspace, outgrads.get(parent), outgrad)
    return cur_outgrad[0]

class VJPNode(Node):
    def process_local_data(self, fun, args, kwargs, ans, argnums):
        self.vspace = vspace(ans)
        self.vjps = [
            primitive_vjp(fun, argnum, ans, vspace(args[argnum]), vspace(ans), args, kwargs)
            for argnum in argnums]

def add_outgrads(vs, prev_g_flagged, g):
    sparse = type(getval(g)) == SparseObject
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

class SparseObject(object):
    __slots__ = ['vs', 'mut_add']
    def __init__(self, vs, mut_add):
        self.vs = vs
        self.mut_add = mut_add
register_vspace(lambda x : x.vs, SparseObject)
register_box(Box, SparseObject)

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

class first_arg_as_get(object):
    def __init__(self, f):
        self.f = f
    def __getitem__(self, argnum):
        def vjp(ans, vs, gvs, args, kwargs):
            return self.f(argnum, ans, vs, gvs, args, kwargs)
        return vjp

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
