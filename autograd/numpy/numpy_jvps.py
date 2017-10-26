from itertools import repeat
from . import numpy_wrapper as anp
from .numpy_vjps import (untake, balanced_eq, match_complex, replace_zero,
                         dot_adjoint_0, dot_adjoint_1, tensordot_adjoint_0,
                         tensordot_adjoint_1, nograd_functions, unbroadcast_f,
                         _broadcast_to_adjoint)
from autograd.extend import (defjvp, defjvp_argnum, def_linear, vspace, JVPNode,
                             register_notrace, defvjp)

from ..util import func, subval
from .numpy_boxes import ArrayBox

for fun in nograd_functions:
    register_notrace(JVPNode, fun)

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

    binary_ufunc_jps = {
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
    if len(derivs_ops) == 2:
        defjvp(ufunc, *[binary_ufunc_jps[op][0](argnum, deriv) for argnum, (deriv, op) in enumerate(derivs_ops)])
        defvjp(ufunc, *[binary_ufunc_jps[op][1](argnum, deriv) for argnum, (deriv, op) in enumerate(derivs_ops)])

defjvp(anp.broadcast_to, 'same')
defjvp(_broadcast_to_adjoint, 'same')

defjvp(func(ArrayBox.__getitem__), 'same')
defjvp(untake, 'same')

defjvp_argnum(anp.array_from_args, lambda argnum, g, ans, args, kwargs: untake(g, argnum-2, vspace(ans)))
defjvp(anp._array_from_scalar_or_array, None, None,
       lambda g, ans, args, kwargs, _: anp._array_from_scalar_or_array(args, kwargs, g))

# ----- Functions that are constant w.r.t. continuous inputs -----
defjvp(anp.nan_to_num, lambda g, ans, x: anp.where(anp.isfinite(x), g, 0.))

# ----- Unary ufuncs ------
def_ufunc_jps(anp.negative,      (None, 'same'))
def_ufunc_jps(anp.rad2deg,       (None, 'same'))
def_ufunc_jps(anp.degrees,       (None, 'same'))
def_ufunc_jps(anp.deg2rad,       (None, 'same'))
def_ufunc_jps(anp.radians,       (None, 'same'))
def_ufunc_jps(anp.abs,
        (lambda ans, x: replace_zero(anp.conj(x), 0.) / replace_zero(ans, 1.), 'cmul'))
def_ufunc_jps(anp.fabs,        (lambda ans, x: anp.sign(x), 'mul'))  # fabs doesn't take complex numbers.
def_ufunc_jps(anp.absolute,    (lambda ans, x: anp.conj(x) / ans,         'cmul'))
def_ufunc_jps(anp.reciprocal,  (lambda ans, x: -ans**2,                   'mul' ))
def_ufunc_jps(anp.exp,         (lambda ans, x: ans,                       'mul' ))
def_ufunc_jps(anp.exp2,        (lambda ans, x: ans * anp.log(2),          'mul' ))
def_ufunc_jps(anp.expm1,       (lambda ans, x: ans + 1,                   'mul' ))
def_ufunc_jps(anp.log,         (lambda ans, x: x,                         'div' ))
def_ufunc_jps(anp.log2,        (lambda ans, x: x * anp.log(2),            'div' ))
def_ufunc_jps(anp.log10,       (lambda ans, x: x * anp.log(10),           'div' ))
def_ufunc_jps(anp.log1p,       (lambda ans, x: x + 1,                     'div' ))
def_ufunc_jps(anp.sin,         (lambda ans, x: anp.cos(x),                'mul' ))
def_ufunc_jps(anp.cos,         (lambda ans, x: -anp.sin(x),               'mul' ))
def_ufunc_jps(anp.tan,         (lambda ans, x: 1 + ans**2,                'mul' ))
def_ufunc_jps(anp.arcsin,      (lambda ans, x: anp.sqrt(1 - x**2),        'div' ))
def_ufunc_jps(anp.arccos,      (lambda ans, x:-anp.sqrt(1 - x**2),        'div' ))
def_ufunc_jps(anp.arctan,      (lambda ans, x: 1 + x**2,                  'div' ))
def_ufunc_jps(anp.sinh,        (lambda ans, x: anp.cosh(x),               'mul' ))
def_ufunc_jps(anp.cosh,        (lambda ans, x: anp.sinh(x),               'mul' ))
def_ufunc_jps(anp.tanh,        (lambda ans, x: 1 - ans**2,                'mul' ))
def_ufunc_jps(anp.arcsinh,     (lambda ans, x: anp.sqrt(x**2 + 1),        'div' ))
def_ufunc_jps(anp.arccosh,     (lambda ans, x: anp.sqrt(x**2 - 1),        'div' ))
def_ufunc_jps(anp.arctanh,     (lambda ans, x: 1 - x**2,                  'div' ))
def_ufunc_jps(anp.square,      (lambda ans, x: 2 * x,                     'mul' ))
def_ufunc_jps(anp.sqrt,        (lambda ans, x: 2 * ans,                   'div' ))
def_ufunc_jps(anp.sinc,        (lambda ans, x: (anp.cos(anp.pi*x)-ans)/x, 'mul' ))
def_ufunc_jps(anp.real_if_close, (None, 'cid'))
def_ufunc_jps(anp.real,        (None, 'cid'))
def_ufunc_jps(anp.imag,        (lambda ans, x: -1j, 'cmul'))
def_ufunc_jps(anp.conj,        (None, 'same'))
def_ufunc_jps(anp.conjugate,   (None, 'same'))
def_ufunc_jps(anp.angle,       (lambda ans, x: anp.conj(x * 1j)/anp.abs(x)**2, 'cmul'))

# ----- Binary ufuncs -----
def_ufunc_jps(anp.add,         *repeat((None, 'id'), 2))
def_ufunc_jps(anp.subtract,    (None, 'id'), (None, 'neg'))
def_ufunc_jps(anp.multiply,    *repeat((None, 'same'), 2))
def_ufunc_jps(anp.divide,      (None, 'same'), (lambda ans, x, y: -ans/y, 'mul'))
def_ufunc_jps(anp.maximum,     (lambda ans, x, y: balanced_eq(x, ans, y), 'mul'),
                               (lambda ans, x, y: balanced_eq(y, ans, x), 'mul'))
def_ufunc_jps(anp.minimum,     (lambda ans, x, y: balanced_eq(x, ans, y), 'mul'),
                               (lambda ans, x, y: balanced_eq(y, ans, x), 'mul'))
def_ufunc_jps(anp.fmax,        (lambda ans, x, y: balanced_eq(x, ans, y), 'mul'),
                               (lambda ans, x, y: balanced_eq(y, ans, x), 'mul'))
def_ufunc_jps(anp.fmin,        (lambda ans, x, y: balanced_eq(x, ans, y), 'mul'),
                               (lambda ans, x, y: balanced_eq(y, ans, x), 'mul'))
def_ufunc_jps(anp.logaddexp,   (lambda ans, x, y: anp.exp(x-ans), 'mul'),
                               (lambda ans, x, y: anp.exp(y-ans), 'mul'))
def_ufunc_jps(anp.logaddexp2,  (lambda ans, x, y: 2**(x-ans), 'mul'),
                               (lambda ans, x, y: 2**(y-ans), 'mul'))
def_ufunc_jps(anp.true_divide, (None, 'same'), (lambda ans, x, y: -ans/y, 'mul'))
def_ufunc_jps(anp.mod,         (None, 'id'), (lambda ans, x, y: -anp.floor(x/y), 'mul'))
def_ufunc_jps(anp.remainder,   (None, 'id'), (lambda ans, x, y: -anp.floor(x/y), 'mul'))
def_ufunc_jps(anp.power,       (lambda ans, x, y: y * x ** anp.where(y, y - 1, 1.), 'mul'),
                               (lambda ans, x, y: anp.log(replace_zero(x, 1.)) * x ** y, 'mul'))

# ----- Simple grads (linear) -----
defjvp(anp.reshape,       'same')
defjvp(anp.roll,          'same')
defjvp(anp.array_split,   'same')
defjvp(anp.split,         'same')
defjvp(anp.vsplit,        'same')
defjvp(anp.hsplit,        'same')
defjvp(anp.dsplit,        'same')
defjvp(anp.ravel,         'same')
defjvp(anp.expand_dims,   'same')
defjvp(anp.squeeze,       'same')
defjvp(anp.diag,          'same')
defjvp(anp.diagonal,      'same')
defjvp(anp.make_diagonal, 'same')
defjvp(anp.flipud,        'same')
defjvp(anp.fliplr,        'same')
defjvp(anp.rot90,         'same')
defjvp(anp.trace,         'same')
defjvp(anp.full,          'same', argnums=(1,))
defjvp(anp.triu,          'same')
defjvp(anp.tril,          'same')
defjvp(anp.swapaxes,      'same')
defjvp(anp.rollaxis,      'same')
defjvp(anp.moveaxis,      'same')
def_linear(anp.cross)

# ----- Simple grads -----
defjvp(anp.clip,        lambda g, ans, x, a_min, a_max : g * anp.logical_and(ans != a_min, ans != a_max))
defjvp(anp.where,  None,
       lambda g, ans, c, x=None, y=None : anp.where(c, g, anp.zeros(anp.shape(g))),
       lambda g, ans, c, x=None, y=None : anp.where(c, anp.zeros(g.shape), g))

# ----- Trickier grads -----
defjvp(anp.kron,      'same', 'same')
defjvp(anp.diff,      'same')
defjvp(anp.repeat,    'same')
defjvp(anp.tile,      'same')
defjvp(anp.transpose, 'same')
defjvp(anp.sum,       'same')
defjvp(anp.mean,      'same')
defjvp(anp.prod, lambda g, ans, x, axis=None, keepdims=False: ans * anp.sum(g / x, axis=axis, keepdims=keepdims))
defjvp(anp.linspace, lambda g, ans, start, stop, *args, **kwargs: anp.linspace(g, 0, *args, **kwargs),
                     lambda g, ans, start, stop, *args, **kwargs: anp.linspace(0, g, *args, **kwargs))

def forward_grad_np_var(g, ans, x, axis=None, ddof=0, keepdims=False):
    if axis is None:
        num_reps = anp.size(g)
    elif isinstance(axis, int):
        num_reps = anp.shape(g)[axis]
    elif isinstance(axis, tuple):
        num_reps = anp.prod(anp.array(np.shape(g))[list(axis)])

    x_minus_mean = anp.conj(x - anp.mean(x, axis=axis, keepdims=True))
    return (2.0 * anp.sum(anp.real(g * x_minus_mean), axis=axis, keepdims=keepdims) /
            (num_reps - ddof))
defjvp(anp.var, forward_grad_np_var)

def forward_grad_np_std(g, ans, x, axis=None, ddof=0, keepdims=False):
    if axis is None:
        num_reps = anp.size(g)
    elif isinstance(axis, int):
        num_reps = anp.shape(g)[axis]
    elif isinstance(axis, tuple):
        num_reps = anp.prod(anp.array(anp.shape(g))[list(axis)])

    if num_reps <= 1:
        return anp.zeros_like(ans)
    x_minus_mean = anp.conj(x - anp.mean(x, axis=axis, keepdims=True))
    return (anp.sum(anp.real(g * x_minus_mean), axis=axis, keepdims=keepdims) /
            ((num_reps - ddof) * ans))
defjvp(anp.std, forward_grad_np_std)

def fwd_grad_chooser(g, ans, x, axis=None, keepdims=False):
    if anp.isscalar(x):
        return g
    if not keepdims:
        if isinstance(axis, int):
            ans = anp.expand_dims(ans, axis)
        elif isinstance(axis, tuple):
            for ax in sorted(axis):
                ans = anp.expand_dims(ans, ax)
    chosen_locations = x == ans
    return (anp.sum((g * chosen_locations), axis=axis, keepdims=keepdims) /
            anp.sum(chosen_locations, axis=axis, keepdims=keepdims))

defjvp(anp.max, fwd_grad_chooser)
defjvp(anp.min, fwd_grad_chooser)
defjvp(anp.amax, fwd_grad_chooser)
defjvp(anp.amin, fwd_grad_chooser)

defjvp(anp.cumsum, 'same')

def_linear(anp.inner)
def_linear(anp.matmul)
def_linear(anp.dot)
def_linear(anp.tensordot)
def_linear(anp.outer)

def_linear(dot_adjoint_0)
def_linear(dot_adjoint_1)

def_linear(tensordot_adjoint_0)
def_linear(tensordot_adjoint_1)

def fwd_grad_concatenate_args(argnum, g, ans, axis_args, kwargs):
    result = []
    for i in range(1, len(axis_args)):
        if i == argnum:
            result.append(g)
        else:
            result.append(anp.zeros_like(axis_args[i]))
    return anp.concatenate_args(axis_args[0], *result)
defjvp_argnum(anp.concatenate_args, fwd_grad_concatenate_args)

def fwd_grad_sort(g, ans, x, axis=-1, kind='quicksort', order=None):
    sort_perm = anp.argsort(x, axis, kind, order)
    return g[sort_perm]
defjvp(anp.sort, fwd_grad_sort)
defjvp(anp.msort, lambda g, ans, x: fwd_grad_sort(g, ans, x, axis=0))

def fwd_grad_partition(g, ans, x, kth, axis=-1, kind='introselect', order=None):
    partition_perm = anp.argpartition(x, kth, axis, kind, order)
    return g[partition_perm]
defjvp(anp.partition, fwd_grad_partition)

def atleast_jvpmaker(fun):
    def jvp(g, ans, *arys):
        if len(arys) > 1:
            raise NotImplementedError("Can't handle multiple arguments yet.")
        return fun(g)
    return jvp
defjvp(anp.atleast_1d, atleast_jvpmaker(anp.atleast_1d))
defjvp(anp.atleast_2d, atleast_jvpmaker(anp.atleast_2d))
defjvp(anp.atleast_3d, atleast_jvpmaker(anp.atleast_3d))

def_linear(anp.einsum)
