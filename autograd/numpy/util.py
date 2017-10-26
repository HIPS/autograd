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
        'same': (lambda deriv: lambda g, ans, x:        ufunc(g),
                 lambda deriv: lambda ans, x: ufunc),
        'mul' : (lambda deriv: lambda g, ans, x:        g * deriv(ans, x),
                 lambda deriv: lambda ans, x: lambda g, d=deriv(ans, x): g * d),
        'div' : (lambda deriv: lambda g, ans, x:        g / deriv(ans, x),
                 lambda deriv: lambda ans, x: lambda g, d=deriv(ans, x): g / d),
        'cmul': (lambda deriv: lambda g, ans, x:        match_complex(ans, g * deriv(ans, x)),
                 lambda deriv: lambda ans, x: lambda g, d=deriv(ans, x): match_complex(x, g * d)),
        'cid':  (lambda deriv: lambda g, ans, x:        match_complex(ans, g),
                 lambda deriv: lambda ans, x: lambda g: match_complex(x  , g))
        }

    if len(derivs_ops) == 1:
        deriv, op = derivs_ops[0]
        defjvp(ufunc, unary_ufunc_jps[op][0](deriv))
        defvjp(ufunc, unary_ufunc_jps[op][1](deriv))

    nary_ufunc_jps = {
        'same': (lambda argnum, deriv: lambda g, ans, *args: ufunc(*subval(args, argnum, g)),
                 lambda argnum, deriv: lambda ans, *args:
                     unbroadcast_f(args[argnum], lambda g: ufunc(*subval(args, argnum, g)))),
        'id':   (lambda argnum, deriv: lambda g, ans, *args: match_complex(ans, anp.broadcast_to(g, ans.shape)),
                 lambda argnum, deriv: lambda ans, *args:
                     unbroadcast_f(args[argnum], lambda g: match_complex(args[argnum], g))),
        'neg':  (lambda argnum, deriv: lambda g, ans, *args: match_complex(ans, anp.broadcast_to(-g, ans.shape)),
                 lambda argnum, deriv: lambda ans, *args:
                     unbroadcast_f(args[argnum], lambda g: match_complex(args[argnum], -g))),
        'mul':  (lambda argnum, deriv: lambda g, ans, *args: g * deriv(ans, *args),
                 lambda argnum, deriv: lambda ans, *args:
                     unbroadcast_f(args[argnum], lambda g, d=deriv(ans, *args): g * d)),
        'div':  (lambda argnum, deriv: lambda g, ans, *args: g / deriv(ans, *args),
                 lambda argnum, deriv: lambda ans, *args:
                     unbroadcast_f(args[argnum], lambda g, d=deriv(ans, *args): g / d))
        }
    if len(derivs_ops) >= 2:
        defjvp(ufunc, *[nary_ufunc_jps[op][0](argnum, deriv) for argnum, (deriv, op) in enumerate(derivs_ops)])
        defvjp(ufunc, *[nary_ufunc_jps[op][1](argnum, deriv) for argnum, (deriv, op) in enumerate(derivs_ops)])

