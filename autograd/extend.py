from functools import partial
from collections import defaultdict
from .util import subval
from .core import defvjp_argnums, defvjp, defjvps, defjvp, defjvp_argnum, def_multilinear
from .vspace import vspace

# Expose API for extending autograd
from .vspace import VSpace, vspace
from .tracer import Box, primitive, notrace_primitive, getval
from .core import SparseObject

# -------------------- reverse mode --------------------

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


# -------------------- forward mode --------------------

def zero_jvp(g, ans, *args, **kwargs): return vspace(ans).zeros()

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
