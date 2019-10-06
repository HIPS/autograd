from . import numpy_wrapper as anp
from .numpy_vjps import (untake, balanced_eq, match_complex, replace_zero,
                         dot_adjoint_0, dot_adjoint_1, tensordot_adjoint_0,
                         tensordot_adjoint_1, nograd_functions)
from autograd.extend import (defjvp, defjvp_argnum, def_linear, vspace, JVPNode,
                             register_notrace)
from ..util import func
from .numpy_boxes import ArrayBox

for fun in nograd_functions:
    register_notrace(JVPNode, fun)

defjvp(func(ArrayBox.__getitem__), 'same')
defjvp(untake, 'same')

defjvp_argnum(anp.array_from_args, lambda argnum, g, ans, args, kwargs: untake(g, argnum-2, vspace(ans)))
defjvp(anp._array_from_scalar_or_array, None, None,
       lambda g, ans, args, kwargs, _: anp._array_from_scalar_or_array(args, kwargs, g))

# ----- Functions that are constant w.r.t. continuous inputs -----
defjvp(anp.nan_to_num, lambda g, ans, x: anp.where(anp.isfinite(x), g, 0.))

# ----- Binary ufuncs (linear) -----
def_linear(anp.multiply)

# ----- Binary ufuncs -----
defjvp(anp.add,        lambda g, ans, x, y : broadcast(g, ans),
                       lambda g, ans, x, y : broadcast(g, ans))
defjvp(anp.subtract,   lambda g, ans, x, y : broadcast(g, ans),
                       lambda g, ans, x, y : broadcast(-g, ans))
defjvp(anp.divide,     'same',
                       lambda g, ans, x, y : - g * x / y**2)
defjvp(anp.maximum,    lambda g, ans, x, y : g * balanced_eq(x, ans, y),
                       lambda g, ans, x, y : g * balanced_eq(y, ans, x))
defjvp(anp.minimum,    lambda g, ans, x, y : g * balanced_eq(x, ans, y),
                       lambda g, ans, x, y : g * balanced_eq(y, ans, x))
defjvp(anp.fmax,       lambda g, ans, x, y : g * balanced_eq(x, ans, y),
                       lambda g, ans, x, y : g * balanced_eq(y, ans, x))
defjvp(anp.fmin,       lambda g, ans, x, y : g * balanced_eq(x, ans, y),
                       lambda g, ans, x, y : g * balanced_eq(y, ans, x))
defjvp(anp.logaddexp,  lambda g, ans, x, y : g * anp.exp(x-ans),
                       lambda g, ans, x, y : g * anp.exp(y-ans))
defjvp(anp.logaddexp2, lambda g, ans, x, y : g * 2**(x-ans),
                       lambda g, ans, x, y : g * 2**(y-ans))
defjvp(anp.true_divide,'same',
                       lambda g, ans, x, y : - g * x / y**2)
defjvp(anp.mod,        lambda g, ans, x, y : broadcast(g, ans),
                       lambda g, ans, x, y : -g * anp.floor(x/y))
defjvp(anp.remainder,  lambda g, ans, x, y : broadcast(g, ans),
                       lambda g, ans, x, y : -g * anp.floor(x/y))
defjvp(anp.power,      lambda g, ans, x, y : g * y * x ** anp.where(y, y - 1, 1.),
                       lambda g, ans, x, y : g * anp.log(replace_zero(x, 1.)) * ans)
defjvp(anp.arctan2,    lambda g, ans, x, y : g * y / (x**2 + y**2),
                       lambda g, ans, x, y : g * -x / (x**2 + y**2))

# ----- Simple grads (linear) -----
defjvp(anp.negative,      'same')
defjvp(anp.rad2deg,       'same')
defjvp(anp.degrees,       'same')
defjvp(anp.deg2rad,       'same')
defjvp(anp.radians,       'same')
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
defjvp(anp.broadcast_to,  'same')
def_linear(anp.cross)

# ----- Simple grads -----
defjvp(anp.abs,
    lambda g, ans, x : anp.real(g * replace_zero(anp.conj(x), 0.)) / replace_zero(ans, 1.))
defjvp(anp.fabs,        lambda g, ans, x : anp.sign(x) * g)  # fabs doesn't take complex numbers.
defjvp(anp.absolute,    lambda g, ans, x : anp.real(g * anp.conj(x)) / ans)
defjvp(anp.reciprocal,  lambda g, ans, x : - g / x**2)
defjvp(anp.exp,         lambda g, ans, x : ans * g)
defjvp(anp.exp2,        lambda g, ans, x : ans * anp.log(2) * g)
defjvp(anp.expm1,       lambda g, ans, x : (ans + 1) * g)
defjvp(anp.log,         lambda g, ans, x : g / x)
defjvp(anp.log2,        lambda g, ans, x : g / x / anp.log(2))
defjvp(anp.log10,       lambda g, ans, x : g / x / anp.log(10))
defjvp(anp.log1p,       lambda g, ans, x : g / (x + 1))
defjvp(anp.sin,         lambda g, ans, x : g * anp.cos(x))
defjvp(anp.cos,         lambda g, ans, x : - g * anp.sin(x))
defjvp(anp.tan,         lambda g, ans, x : g / anp.cos(x) **2)
defjvp(anp.arcsin,      lambda g, ans, x : g / anp.sqrt(1 - x**2))
defjvp(anp.arccos,      lambda g, ans, x :-g / anp.sqrt(1 - x**2))
defjvp(anp.arctan,      lambda g, ans, x : g / (1 + x**2))
defjvp(anp.sinh,        lambda g, ans, x : g * anp.cosh(x))
defjvp(anp.cosh,        lambda g, ans, x : g * anp.sinh(x))
defjvp(anp.tanh,        lambda g, ans, x : g / anp.cosh(x) **2)
defjvp(anp.arcsinh,     lambda g, ans, x : g / anp.sqrt(x**2 + 1))
defjvp(anp.arccosh,     lambda g, ans, x : g / anp.sqrt(x**2 - 1))
defjvp(anp.arctanh,     lambda g, ans, x : g / (1 - x**2))
defjvp(anp.square,      lambda g, ans, x : g * 2 * x)
defjvp(anp.sqrt,        lambda g, ans, x : g * 0.5 * x**-0.5)
defjvp(anp.sinc,        lambda g, ans, x : g * (anp.cos(anp.pi*x)*anp.pi*x - anp.sin(anp.pi*x))/(anp.pi*x**2))
defjvp(anp.clip,        lambda g, ans, x, a_min, a_max : g * anp.logical_and(ans != a_min, ans != a_max))
defjvp(anp.real_if_close, lambda g, ans, x : match_complex(ans, g))
defjvp(anp.real,   lambda g, ans, x   : anp.real(g))
defjvp(anp.imag,   lambda g, ans, x   : match_complex(ans, -1j * g))
defjvp(anp.conj,   lambda g, ans, x   : anp.conj(g))
defjvp(anp.angle,  lambda g, ans, x   : match_complex(ans, g * anp.conj(x * 1j) / anp.abs(x)**2))
defjvp(anp.where,  None,
       lambda g, ans, c, x=None, y=None : anp.where(c, g, anp.zeros(anp.shape(g))),
       lambda g, ans, c, x=None, y=None : anp.where(c, anp.zeros(g.shape), g))

# ----- Trickier grads -----
defjvp(anp.kron,      'same', 'same')
defjvp(anp.diff,      'same')
defjvp(anp.gradient,  'same')
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

# TODO(mattjj): can we call np.broadcast_to or a related function instead?
def broadcast(x, target):
    target_shape, target_ndim, target_dtype, target_iscomplex = anp.metadata(target)
    while anp.ndim(x) < target_ndim:
        x = anp.expand_dims(x, 0)
    for axis, size in enumerate(anp.shape(x)):
        if size == 1:
            x = anp.repeat(x, target_shape[axis], axis=axis)
    if target_iscomplex and not anp.iscomplexobj(x):
        x = x + 0j  # TODO(mattjj): this might promote the dtype
    return x

defjvp(anp.pad, lambda g, ans, array, width, mode, **kwargs:
       anp.pad(g, width, mode))
