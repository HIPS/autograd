from . import numpy_wrapper as anp
from .numpy_vjps import (untake, balanced_eq, match_complex, replace_zero,
                         dot_0_adjoint, dot_1_adjoint)
from autograd.core import (defjvp, defjvps, def_linear_wrt_arg, defjvp_argnum,
                           def_multilinear, vspace)
from ..util import func
from .numpy_boxes import ArrayBox

def_linear_wrt_arg(func(ArrayBox.__getitem__))
def_linear_wrt_arg(untake)

defjvp_argnum(anp.array_from_args, lambda argnum, g, ans, gvs, vs, *args: untake(g, argnum, vs))

# ----- Functions that are constant w.r.t. continuous inputs -----
defjvp(anp.nan_to_num, lambda g, ans, gvs, vs, x: anp.where(anp.isfinite(x), g, 0.))

# ----- Binary ufuncs (linear) -----
def_multilinear(anp.multiply)
def_linear_wrt_arg(anp.divide)
def_linear_wrt_arg(anp.true_divide)

# ----- Binary ufuncs -----
defjvp(anp.add,        lambda g, ans, gvs, vs, x, y : broadcast(gvs, vs, g))
defjvp(anp.add,        lambda g, ans, gvs, vs, x, y : broadcast(gvs, vs, g), argnum=1)
defjvp(anp.subtract,   lambda g, ans, gvs, vs, x, y : broadcast(gvs, vs, g))
defjvp(anp.subtract,   lambda g, ans, gvs, vs, x, y : broadcast(gvs, vs, -g), argnum=1)
defjvp(anp.divide,     lambda g, ans, gvs, vs, x, y : - g * x / y**2, argnum=1)
defjvp(anp.maximum,    lambda g, ans, gvs, vs, x, y : g * balanced_eq(x, ans, y))
defjvp(anp.maximum,    lambda g, ans, gvs, vs, x, y : g * balanced_eq(y, ans, x), argnum=1)
defjvp(anp.minimum,    lambda g, ans, gvs, vs, x, y : g * balanced_eq(x, ans, y))
defjvp(anp.minimum,    lambda g, ans, gvs, vs, x, y : g * balanced_eq(y, ans, x), argnum=1)
defjvp(anp.fmax,       lambda g, ans, gvs, vs, x, y : g * balanced_eq(x, ans, y))
defjvp(anp.fmax,       lambda g, ans, gvs, vs, x, y : g * balanced_eq(y, ans, x), argnum=1)
defjvp(anp.fmin,       lambda g, ans, gvs, vs, x, y : g * balanced_eq(x, ans, y))
defjvp(anp.fmin,       lambda g, ans, gvs, vs, x, y : g * balanced_eq(y, ans, x), argnum=1)
defjvp(anp.logaddexp,  lambda g, ans, gvs, vs, x, y : g * anp.exp(x-ans))
defjvp(anp.logaddexp,  lambda g, ans, gvs, vs, x, y : g * anp.exp(y-ans), argnum=1)
defjvp(anp.logaddexp2, lambda g, ans, gvs, vs, x, y : g * 2**(x-ans))
defjvp(anp.logaddexp2, lambda g, ans, gvs, vs, x, y : g * 2**(y-ans), argnum=1)
defjvp(anp.true_divide,lambda g, ans, gvs, vs, x, y : - g * x / y**2, argnum=1)
defjvp(anp.mod,        lambda g, ans, gvs, vs, x, y : broadcast(gvs, vs, g))
defjvp(anp.remainder,  lambda g, ans, gvs, vs, x, y : broadcast(gvs, vs, g))
defjvp(anp.mod,        lambda g, ans, gvs, vs, x, y : -g * anp.floor(x/y), argnum=1)
defjvp(anp.remainder,  lambda g, ans, gvs, vs, x, y : -g * anp.floor(x/y), argnum=1)
defjvp(anp.power,      lambda g, ans, gvs, vs, x, y : g * y * x ** anp.where(y, y - 1, 1.))
defjvp(anp.power,      lambda g, ans, gvs, vs, x, y : g * anp.log(replace_zero(x, 1.)) * x ** y, argnum=1)

# ----- Simple grads (linear) -----
def_linear_wrt_arg(anp.negative)
def_linear_wrt_arg(anp.rad2deg)
def_linear_wrt_arg(anp.degrees)
def_linear_wrt_arg(anp.deg2rad)
def_linear_wrt_arg(anp.radians)
def_linear_wrt_arg(anp.reshape)
def_linear_wrt_arg(anp.roll)
def_linear_wrt_arg(anp.array_split)
def_linear_wrt_arg(anp.split)
def_linear_wrt_arg(anp.vsplit)
def_linear_wrt_arg(anp.hsplit)
def_linear_wrt_arg(anp.dsplit)
def_linear_wrt_arg(anp.ravel)
def_linear_wrt_arg(anp.expand_dims)
def_linear_wrt_arg(anp.squeeze)
def_linear_wrt_arg(anp.diag)
def_linear_wrt_arg(anp.diagonal)
def_linear_wrt_arg(anp.make_diagonal)
def_linear_wrt_arg(anp.flipud)
def_linear_wrt_arg(anp.fliplr)
def_linear_wrt_arg(anp.rot90)
def_linear_wrt_arg(anp.trace)
def_linear_wrt_arg(anp.full, argnum=1)
def_linear_wrt_arg(anp.triu)
def_linear_wrt_arg(anp.tril)
def_linear_wrt_arg(anp.swapaxes)
def_linear_wrt_arg(anp.rollaxis)
def_linear_wrt_arg(anp.moveaxis)
def_multilinear(anp.cross)

# ----- Simple grads -----
defjvp(anp.abs,
    lambda g, ans, gvs, vs, x : anp.real(g * replace_zero(anp.conj(x), 0.)) / replace_zero(ans, 1.))
defjvp(anp.fabs,        lambda g, ans, gvs, vs, x : anp.sign(x) * g)  # fabs doesn't take complex numbers.
defjvp(anp.absolute,    lambda g, ans, gvs, vs, x : anp.real(g * anp.conj(x)) / ans)
defjvp(anp.reciprocal,  lambda g, ans, gvs, vs, x : - g / x**2)
defjvp(anp.exp,         lambda g, ans, gvs, vs, x : ans * g)
defjvp(anp.exp2,        lambda g, ans, gvs, vs, x : ans * anp.log(2) * g)
defjvp(anp.expm1,       lambda g, ans, gvs, vs, x : (ans + 1) * g)
defjvp(anp.log,         lambda g, ans, gvs, vs, x : g / x)
defjvp(anp.log2,        lambda g, ans, gvs, vs, x : g / x / anp.log(2))
defjvp(anp.log10,       lambda g, ans, gvs, vs, x : g / x / anp.log(10))
defjvp(anp.log1p,       lambda g, ans, gvs, vs, x : g / (x + 1))
defjvp(anp.sin,         lambda g, ans, gvs, vs, x : g * anp.cos(x))
defjvp(anp.cos,         lambda g, ans, gvs, vs, x : - g * anp.sin(x))
defjvp(anp.tan,         lambda g, ans, gvs, vs, x : g / anp.cos(x) **2)
defjvp(anp.arcsin,      lambda g, ans, gvs, vs, x : g / anp.sqrt(1 - x**2))
defjvp(anp.arccos,      lambda g, ans, gvs, vs, x :-g / anp.sqrt(1 - x**2))
defjvp(anp.arctan,      lambda g, ans, gvs, vs, x : g / (1 + x**2))
defjvp(anp.sinh,        lambda g, ans, gvs, vs, x : g * anp.cosh(x))
defjvp(anp.cosh,        lambda g, ans, gvs, vs, x : g * anp.sinh(x))
defjvp(anp.tanh,        lambda g, ans, gvs, vs, x : g / anp.cosh(x) **2)
defjvp(anp.arcsinh,     lambda g, ans, gvs, vs, x : g / anp.sqrt(x**2 + 1))
defjvp(anp.arccosh,     lambda g, ans, gvs, vs, x : g / anp.sqrt(x**2 - 1))
defjvp(anp.arctanh,     lambda g, ans, gvs, vs, x : g / (1 - x**2))
defjvp(anp.square,      lambda g, ans, gvs, vs, x : g * 2 * x)
defjvp(anp.sqrt,        lambda g, ans, gvs, vs, x : g * 0.5 * x**-0.5)
defjvp(anp.sinc,        lambda g, ans, gvs, vs, x : g * (anp.cos(anp.pi*x)*anp.pi*x - anp.sin(anp.pi*x))/(anp.pi*x**2))
defjvp(anp.clip,        lambda g, ans, gvs, vs, x, a_min, a_max : g * anp.logical_and(ans != a_min, ans != a_max))
defjvp(anp.real_if_close, lambda g, ans, gvs, vs, x : match_complex(vs, g))
defjvp(anp.real,   lambda g, ans, gvs, vs, x   : anp.real(g))
defjvp(anp.imag,   lambda g, ans, gvs, vs, x   : match_complex(vs, -1j * g))
defjvp(anp.conj,   lambda g, ans, gvs, vs, x   : anp.conj(g))
defjvp(anp.angle,  lambda g, ans, gvs, vs, x   : match_complex(vs, g * anp.conj(x * 1j) / anp.abs(x)**2))
defjvp(anp.where,  lambda g, ans, gvs, vs, c, x=None, y=None : anp.where(c, g, anp.zeros(anp.shape(g))), argnum=1)
defjvp(anp.where,  lambda g, ans, gvs, vs, c, x=None, y=None : anp.where(c, anp.zeros(g.shape), g), argnum=2)

# ----- Trickier grads -----
def_linear_wrt_arg(anp.diff)
def_linear_wrt_arg(anp.repeat)
def_linear_wrt_arg(anp.tile)
def_multilinear(anp.kron)
def_linear_wrt_arg(anp.transpose)
def_linear_wrt_arg(anp.sum)
def_linear_wrt_arg(anp.mean)
defjvp(anp.prod, lambda g, ans, gvs, vs, x, axis=None, keepdims=False: ans * anp.sum(g / x, axis=axis, keepdims=keepdims))
defjvp(anp.linspace, lambda g, ans, gvs, vs, start, stop, *args, **kwargs: anp.linspace(g, 0, *args, **kwargs), argnum=0)
defjvp(anp.linspace, lambda g, ans, gvs, vs, start, stop, *args, **kwargs: anp.linspace(0, g, *args, **kwargs), argnum=1)

def forward_grad_np_var(g, ans, gvs, vs, x, axis=None, ddof=0, keepdims=False):
    if axis is None:
        if gvs.iscomplex:
            num_reps = gvs.size / 2
        else:
            num_reps = gvs.size
    elif isinstance(axis, int):
        num_reps = gvs.shape[axis]
    elif isinstance(axis, tuple):
        num_reps = anp.prod(anp.array(gvs.shape)[list(axis)])

    x_minus_mean = anp.conj(x - anp.mean(x, axis=axis, keepdims=True))
    return (2.0 * anp.sum(anp.real(g * x_minus_mean), axis=axis, keepdims=keepdims) /
            (num_reps - ddof))
defjvp(anp.var, forward_grad_np_var)

def forward_grad_np_std(g, ans, gvs, vs, x, axis=None, ddof=0, keepdims=False):
    if axis is None:
        if gvs.iscomplex:
            num_reps = gvs.size / 2
        else:
            num_reps = gvs.size
    elif isinstance(axis, int):
        num_reps = gvs.shape[axis]
    elif isinstance(axis, tuple):
        num_reps = anp.prod(anp.array(gvs.shape)[list(axis)])

    if num_reps <= 1:
        return vs.zeros()
    x_minus_mean = anp.conj(x - anp.mean(x, axis=axis, keepdims=True))
    return (anp.sum(anp.real(g * x_minus_mean), axis=axis, keepdims=keepdims) /
            ((num_reps - ddof) * ans))
defjvp(anp.std, forward_grad_np_std)

def fwd_grad_chooser(g, ans, gvs, vs, x, axis=None, keepdims=False):
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

defjvp(anp.cumsum, lambda g, ans, gvs, vs, x, axis=None: anp.cumsum(g, axis=axis))

def_multilinear(anp.inner)
def_multilinear(anp.matmul)
def_multilinear(anp.dot)
def_multilinear(anp.tensordot)
def_multilinear(anp.outer)

def_multilinear(dot_0_adjoint)
def_multilinear(dot_1_adjoint)

def fwd_grad_concatenate_args(argnum, g, ans, gvs, vs, *axis_args, **kwargs):
    result = []
    for i in range(1, len(axis_args)):
        if i == argnum:
            result.append(g)
        else:
            result.append(vspace(axis_args[i]).zeros())
    return anp.concatenate_args(axis_args[0], *result)
defjvp_argnum(anp.concatenate_args, fwd_grad_concatenate_args)

def fwd_grad_sort(g, ans, gvs, vs, x, axis=-1, kind='quicksort', order=None):
    sort_perm = anp.argsort(x, axis, kind, order)
    return g[sort_perm]
defjvp(anp.sort, fwd_grad_sort)
defjvp(anp.msort, lambda g, ans, gvs, vs, x: fwd_grad_sort(g, ans, gvs, vs, x,
                                                          axis=0))

def fwd_grad_partition(g, ans, gvs, vs, x, kth, axis=-1, kind='introselect', order=None):
    partition_perm = anp.argpartition(x, kth, axis, kind, order)
    return g[partition_perm]
defjvp(anp.partition, fwd_grad_partition)

def atleast_jvpmaker(fun):
    def jvp(g, ans, gvs, vs, *arys):
        if len(arys) > 1:
            raise NotImplementedError("Can't handle multiple arguments yet.")
        return fun(g)
    return jvp
defjvp(anp.atleast_1d, atleast_jvpmaker(anp.atleast_1d))
defjvp(anp.atleast_2d, atleast_jvpmaker(anp.atleast_2d))
defjvp(anp.atleast_3d, atleast_jvpmaker(anp.atleast_3d))

def_multilinear(anp.einsum)

def broadcast(gvs, vs, result, broadcast_idx=0):
    while anp.ndim(result) < len(vs.shape):
        result = anp.expand_dims(result, 0)
    for axis, size in enumerate(anp.shape(result)):
        if size == 1:
            result = anp.repeat(result, vs.shape[axis], axis=axis)
    if vs.iscomplex and not gvs.iscomplex:
        result = result + 0j
    return result
