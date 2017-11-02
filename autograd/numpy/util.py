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

def def_unary_ufunc_jps(ufunc, deriv_op):
    jps = {
        'same': (lambda g, ans, x:        ufunc(g),
                 lambda ans, x: ufunc),
        'cid':  (lambda g, ans, x:        match_complex(ans, g),
                 lambda ans, x: lambda g: match_complex(x  , g))
        }

    linops = {
        'mul' : (lambda deriv: lambda g, ans, x:        g * deriv(ans, x),
                 lambda deriv: lambda ans, x: lambda g, d=deriv(ans, x): g * d),
        'div' : (lambda deriv: lambda g, ans, x:        g / deriv(ans, x),
                 lambda deriv: lambda ans, x: lambda g, d=deriv(ans, x): g / d),
        'cmul': (lambda deriv: lambda g, ans, x:        match_complex(ans, g * deriv(ans, x)),
                 lambda deriv: lambda ans, x: lambda g, d=deriv(ans, x): match_complex(x, g * d)),
        }

    if type(deriv_op) is tuple:
        deriv, op = deriv_op
        defjvp(ufunc, linops[op][0](deriv))
        defvjp(ufunc, linops[op][1](deriv))
    elif deriv_op is None:
        defjvp(ufunc, None)
        defvjp(ufunc, None)
    else:
        defjvp(ufunc, jps[deriv_op][0])
        defvjp(ufunc, jps[deriv_op][1])

def def_nary_ufunc_jps(ufunc, derivs_ops):
    jps = {
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

    linops = {
        'mul':  (lambda argnum, deriv: lambda g, ans, *args: g * deriv(ans, *args),
                 lambda argnum, deriv: lambda ans, *args:
                     unbroadcast_f(args[argnum], lambda g, d=deriv(ans, *args): g * d)),
        'div':  (lambda argnum, deriv: lambda g, ans, *args: g / deriv(ans, *args),
                 lambda argnum, deriv: lambda ans, *args:
                 unbroadcast_f(args[argnum], lambda g, d=deriv(ans, *args): g / d))
        }

    def deriv_op_to_jp(idx, argnum, deriv_op):
        if type(deriv_op) is tuple:
            deriv, op = deriv_op
            return linops[op][idx](argnum, deriv)
        elif deriv_op is None:
            return None
        else:
            return jps[deriv_op][idx](argnum)

    defjvp(ufunc, *[deriv_op_to_jp(0, argnum, deriv_op)
                    for argnum, deriv_op in enumerate(derivs_ops)])
    defvjp(ufunc, *[deriv_op_to_jp(1, argnum, deriv_op)
                    for argnum, deriv_op in enumerate(derivs_ops)])

def def_ufunc_jps(ufunc, *derivs_ops):
    """
    Specify the derivatives of ufunc. Once this has been done the ufunc will
    support both reverse and forward mode differentiation.

    The derivatives can be specified as follows.

    Unary ufuncs
    ------------
    If the ufunc is unary (that is, if it takes one array valued argument),
    then a single optional argument is required to specify the ufunc's
    derivative.
    
    In the general case, this is done via a pair (deriv, op), where deriv is a
    function taking in the output of the ufunc (ans), and its array argument
    (x), and returning the derivative of the ufunc.
    
    Here 'derivative' means the elementwise derivative of the ufunc w.r.t. it's
    input.

    For example, for the ufunc np.sin, this is as simple as
    >>> def deriv(ans, x):
    ...     return np.cos(x)
    ...

    Sometimes the output of the ufunc is useful, for example the derivative of
    np.exp is np.exp, which is identical to ans, so the derivative of np.exp
    can be efficiently implemented as
    >>> def deriv(ans, x):
    ...     return ans
    ...

    The other element of the pair is `op`, which should usually be set to
    'mul'. However, if the derivative of the ufunc is of the form
    1 / f(ans, x), then you can save some computation by using the pair
    (f, 'div') to specify the derivative. The 'div' flags that the gradients
    being propagated through this primitive should be divided by the result of
    f, not multiplied.

    Some full examples:
    >>> def_ufunc_jps(np.sin, (lambda ans, x: np.cos(x), 'mul'))
    >>> def_ufunc_jps(np.exp, (lambda ans, x: ans,       'mul'))
    >>> def_ufunc_jps(np.log, (lambda ans, x: x,         'div'))

    Special cases
    -------------
    If the derivative of the ufunc is a constant, then you don't need to
    specify its derivative and you can use just the string 'same' in place of
    the pair (deriv, op). This says that its ok to propagate the gradient
    through this primitive by applying the ufunc itself to the gradient, and
    neither x nor ans are relevant to this computation. 

    For example, the derivative of np.negative (which simply negates its
    inputs), is -1, so
    >>> def_ufunc_jps(np.negative, 'same')

    will correctly set its derivative.

    N-ary ufuncs
    ------------
    For ufuncs which take more than one array argument, the derivatives can be
    specified by passing one (deriv, op) pair for each argument (you can use
    None as a placeholder for args whose derivative you don't wish to define).

    You can use 'same' in exactly the same way as for unary ufuncs, and
    additionally you can use 'id' when the derivative w.r.t. an arg is always
    equal to 1, and 'neg' when it's always equal to -1.

    Some examples:
    >>> def_ufunc_jps(anp.divide,   'same', (lambda ans, x, y: -ans/y, 'mul'))
    >>> def_ufunc_jps(anp.add,      'id', 'id')
    >>> def_ufunc_jps(anp.subtract, 'id', 'neg')
    """
    derivs_ops = list(derivs_ops)
    if len(derivs_ops) == 1:
        def_unary_ufunc_jps(ufunc, derivs_ops[0])
    elif len(derivs_ops) > 1:
        def_nary_ufunc_jps(ufunc, derivs_ops)

def def_ufunc_jps_inv_pair(ufunc, ufunc_inv, deriv):
    """
    Define the derivatives for an inverse pair of unary ufuncs. deriv must be
    the derivative of the first ufunc.
    """
    def_ufunc_jps(ufunc, (deriv, 'mul'))
    def_ufunc_jps(ufunc_inv, (lambda ans, x: deriv(x, ans), 'div'))
