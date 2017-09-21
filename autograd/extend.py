from functools import partial
from collections import defaultdict
from .util import subval
from .core import defvjp_argnums, defjvp, defjvp_argnums, sum_outgrads

# Expose API for extending autograd
from .tracer import Box, primitive, notrace_primitive, getval
from .core import SparseObject, VSpace, vspace

# -------------------- reverse mode defgrad wrappers --------------------

primitive_vjps_onearg = defaultdict(dict)
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

    defvjp_argnums(fun, vjp_argnums)

def defvjp_argnum(fun, vjpmaker):
    def vjp_argnums(argnums, *args):
        vjps = [vjpmaker(argnum, *args) for argnum in argnums]
        return lambda g: (vjp(g) for vjp in vjps)
    defvjp_argnums(fun, vjp_argnums)

def defvjps(fun, vjpmaker, argnums):
    for argnum in argnums:
        defvjp(fun, partial(vjpmaker, argnum), argnum)

def defvjp_is_zero(fun, argnums=(0,)):
    for argnum in argnums:
        defvjp(fun, zero_vjp(argnum), argnum)
        defjvp(fun, zero_jvp, argnum)

def zero_vjp(argnum):
    return lambda ans, *args, **kwargs: lambda g: vspace(args[argnum]).zeros()

# -------------------- forward mode defgrad wrappers  --------------------

def defjvps(fun, jvpfun, argnums):
    for argnum in argnums:
        defjvp(fun, partial(jvpfun, argnum), argnum)

def defjvp_argnum(fun, jvpmaker):
    def jvp_argnums(argnums, gs, ans, args, kwargs):
        return sum_outgrads(jvpmaker(argnum, g, ans, args, kwargs)
                            for argnum, g in zip(argnums, gs))
    defjvp_argnums(fun, jvp_argnums)

def def_multilinear(fun):
    """Flags that a function is linear in all of its args."""
    defjvp_argnum(fun, lambda argnum, g, ans, args, kwargs:
                  fun(*subval(args, argnum, g), **kwargs))

def def_linear_wrt_arg(fun, argnum=0):
    """
    This signifies that a function is linear in the sense of linear
    algebra/functional analysis: fun(a*x + b*y) = a*fun(x) + b*fun(y)
    """
    defjvp(fun, lambda g, ans, *args, **kwargs:
           fun(*subval(args, argnum, g), **kwargs), argnum=argnum)

def def_linear_wrt_args(fun, argnums):
    for argnum in argnums:
        def_linear_wrt_arg(fun, argnum)

def zero_jvp(g, ans, *args, **kwargs): return vspace(ans).zeros()
