from . import numpy_wrapper as anp
from .numpy_vjps import (untake, balanced_eq, match_complex, replace_zero,
                         dot_adjoint_0, dot_adjoint_1, tensordot_adjoint_0,
                         tensordot_adjoint_1, nograd_functions)
from autograd.extend import (defjvp, def_linear, vspace,
                             defjvp_is_fun, defjvp_full, defjvp_zero)
from ..fmap_util import container_fmap, select_map, compose_fmaps
from ..util import func
from .numpy_boxes import ArrayBox

for fun in nograd_functions:
    defjvp_zero(fun)

defjvp_is_fun(func(ArrayBox.__getitem__))
defjvp_is_fun(untake)

# ----- Functions that are constant w.r.t. continuous inputs -----
defjvp(anp.nan_to_num, lambda g, ans, x: anp.where(anp.isfinite(x), g, 0.))

# ----- Binary ufuncs (linear) -----
def_linear(anp.multiply)

# ----- Binary ufuncs -----
defjvp(anp.add,        lambda g, ans, x, y : broadcast(g, ans),
                       lambda g, ans, x, y : broadcast(g, ans))
defjvp(anp.subtract,   lambda g, ans, x, y : broadcast(g, ans),
                       lambda g, ans, x, y : broadcast(-g, ans))
defjvp(anp.divide,     lambda g, ans, x, y : anp.divide(g, y),
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
defjvp(anp.true_divide,lambda g, ans, x, y : anp.true_divide(g, y),
                       lambda g, ans, x, y : - g * x / y**2)
defjvp(anp.mod,        lambda g, ans, x, y : broadcast(g, ans),
                       lambda g, ans, x, y : -g * anp.floor(x/y))
defjvp(anp.remainder,  lambda g, ans, x, y : broadcast(g, ans),
                       lambda g, ans, x, y : -g * anp.floor(x/y))
defjvp(anp.power,      lambda g, ans, x, y : g * y * x ** anp.where(y, y - 1, 1.),
                       lambda g, ans, x, y : g * anp.log(replace_zero(x, 1.)) * x ** y)
defjvp(anp.arctan2,    lambda g, ans, x, y : g * y / (x**2 + y**2),
                       lambda g, ans, x, y : g * -x / (x**2 + y**2))

# ----- Simple grads (linear) -----
defjvp_is_fun(anp.negative)
defjvp_is_fun(anp.rad2deg)
defjvp_is_fun(anp.degrees)
defjvp_is_fun(anp.deg2rad)
defjvp_is_fun(anp.radians)
defjvp_is_fun(anp.reshape)
defjvp_is_fun(anp.roll)
defjvp_is_fun(anp.array_split)
defjvp_is_fun(anp.split)
defjvp_is_fun(anp.vsplit)
defjvp_is_fun(anp.hsplit)
defjvp_is_fun(anp.dsplit)
defjvp_is_fun(anp.ravel)
defjvp_is_fun(anp.expand_dims)
defjvp_is_fun(anp.squeeze)
defjvp_is_fun(anp.diag)
defjvp_is_fun(anp.diagonal)
defjvp_is_fun(anp.make_diagonal)
defjvp_is_fun(anp.flipud)
defjvp_is_fun(anp.fliplr)
defjvp_is_fun(anp.rot90)
defjvp_is_fun(anp.trace)
defjvp_is_fun(anp.full)
defjvp_is_fun(anp.triu)
defjvp_is_fun(anp.tril)
defjvp_is_fun(anp.swapaxes)
defjvp_is_fun(anp.rollaxis)
defjvp_is_fun(anp.moveaxis)
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
defjvp(anp.where,
       lambda g, ans, c, x=None, y=None : anp.zeros(anp.shape(c)),
       lambda g, ans, c, x=None, y=None : anp.where(c, g, anp.zeros(anp.shape(g))),
       lambda g, ans, c, x=None, y=None : anp.where(c, anp.zeros(g.shape), g))

# ----- Trickier grads -----
def_linear(anp.kron)
defjvp_is_fun(anp.diff)
defjvp_is_fun(anp.repeat)
defjvp_is_fun(anp.tile)
defjvp_is_fun(anp.transpose)
defjvp_is_fun(anp.sum)
defjvp_is_fun(anp.mean)
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

defjvp_is_fun(anp.cumsum)

def_linear(anp.inner)
def_linear(anp.matmul)
def_linear(anp.dot)
def_linear(anp.tensordot)
def_linear(anp.outer)

def_linear(dot_adjoint_0)
def_linear(dot_adjoint_1)

def_linear(tensordot_adjoint_0)
def_linear(tensordot_adjoint_1)

def concatenate_jvp(parents, gs, ans, arrs, axis=0):
    result = []
    for arr, g, parent in zip(arrs, gs[0], parents[0]):
        if parent:
            result.append(g)
        else:
            result.append(anp.zeros_like(arr))
    return anp.concatenate(result, axis)

defjvp_full(anp.concatenate, concatenate_jvp)

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

container_fmap_fst = compose_fmaps(select_map([0]), container_fmap)
defjvp_is_fun(anp._array_from_arrays, container_fmap_fst)
defjvp_is_fun(anp._array_from_scalar_or_array)
