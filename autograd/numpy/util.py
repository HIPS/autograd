from . import numpy_wrapper as anp
from autograd.core import defjvp, defvjp
from autograd.util import subval

def match_complex(target, x):
    target_iscomplex = anp.iscomplexobj(target)
    x_iscomplex      = anp.iscomplexobj(x)
    if x_iscomplex and not target_iscomplex:
        return anp.real(x)
    elif not x_iscomplex and target_iscomplex:
        return x + 0j
    else:
        return x

def unbroadcast(x, target_meta, broadcast_idx=0):
    target_shape, target_ndim, dtype, target_iscomplex = target_meta
    x = anp._broadcast_to_adjoint(x, target_shape)
    if anp.iscomplexobj(x) and not target_iscomplex:
        x = anp.real(x)
    return x

def unbroadcast_f(target, f):
    target_meta = anp.metadata(target)
    return lambda g: unbroadcast(f(g), target_meta)

def def_ufunc_jps(ufunc, *derivs_ops):
    derivs_ops = list(derivs_ops)

    unary_ufunc_jps = {
        'same': (lambda g, ans, x:        ufunc(g),
                 lambda ans, x: ufunc),
        'cid':  (lambda g, ans, x:        match_complex(ans, g),
                 lambda ans, x: lambda g: match_complex(x  , g))
        }

    unary_ufunc_linops = {
        'mul' : (lambda deriv: lambda g, ans, x:        g * deriv(ans, x),
                 lambda deriv: lambda ans, x: lambda g, d=deriv(ans, x): g * d),
        'div' : (lambda deriv: lambda g, ans, x:        g / deriv(ans, x),
                 lambda deriv: lambda ans, x: lambda g, d=deriv(ans, x): g / d),
        'cmul': (lambda deriv: lambda g, ans, x:        match_complex(ans, g * deriv(ans, x)),
                 lambda deriv: lambda ans, x: lambda g, d=deriv(ans, x): match_complex(x, g * d)),
        }

    if len(derivs_ops) == 1:
        deriv_op = derivs_ops[0]
        if type(deriv_op) is tuple:
            deriv, op = deriv_op
            defjvp(ufunc, unary_ufunc_linops[op][0](deriv))
            defvjp(ufunc, unary_ufunc_linops[op][1](deriv))
        elif deriv_op is None:
            defjvp(ufunc, None)
            defvjp(ufunc, None)
        else:
            defjvp(ufunc, unary_ufunc_jps[deriv_op][0])
            defvjp(ufunc, unary_ufunc_jps[deriv_op][1])

    nary_ufunc_jps = {
        'same': (lambda argnum: lambda g, ans, *args: ufunc(*subval(args, argnum, g)),
                 lambda argnum: lambda ans, *args:
                     unbroadcast_f(args[argnum], lambda g: ufunc(*subval(args, argnum, g)))),
        'id':   (lambda argnum: lambda g, ans, *args: match_complex(ans, anp.broadcast_to(g, ans.shape)),
                 lambda argnum: lambda ans, *args:
                     unbroadcast_f(args[argnum], lambda g: match_complex(args[argnum], g))),
        'neg':  (lambda argnum: lambda g, ans, *args: match_complex(ans, anp.broadcast_to(-g, ans.shape)),
                 lambda argnum: lambda ans, *args:
                     unbroadcast_f(args[argnum], lambda g: match_complex(args[argnum], -g)))
        }

    nary_ufunc_linops = {
        'mul':  (lambda argnum, deriv: lambda g, ans, *args: g * deriv(ans, *args),
                 lambda argnum, deriv: lambda ans, *args:
                     unbroadcast_f(args[argnum], lambda g, d=deriv(ans, *args): g * d)),
        'div':  (lambda argnum, deriv: lambda g, ans, *args: g / deriv(ans, *args),
                 lambda argnum, deriv: lambda ans, *args:
                     unbroadcast_f(args[argnum], lambda g, d=deriv(ans, *args): g / d))
        }

    def deriv_op_to_ufunc_jp(idx, argnum, deriv_op):
        if type(deriv_op) is tuple:
            deriv, op = deriv_op
            return nary_ufunc_linops[op][idx](argnum, deriv)
        elif deriv_op is None:
            return None
        else:
            return nary_ufunc_jps[deriv_op][idx](argnum)

    if len(derivs_ops) >= 2:
        defjvp(ufunc, *[deriv_op_to_ufunc_jp(0, argnum, deriv_op) for argnum, deriv_op in enumerate(derivs_ops)])
        defvjp(ufunc, *[deriv_op_to_ufunc_jp(1, argnum, deriv_op) for argnum, deriv_op in enumerate(derivs_ops)])
