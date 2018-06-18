from . import cupy_wrapper as acp
from .cupy_vjps import (
    untake,
    balanced_eq,
    match_complex,
    replace_zero,
    dot_adjoint_0,
    dot_adjoint_1,
    tensordot_adjoint_0,
    tensordot_adjoint_1,
    nograd_functions,
)
from autograd.extend import (
    defjvp, defjvp_argnum, def_linear, vspace, JVPNode, register_notrace
)
from ..util import func
from .cupy_boxes import ArrayBox

for fun in nograd_functions:
    register_notrace(JVPNode, fun)

defjvp(func(ArrayBox.__getitem__), "same")
defjvp(untake, "same")

defjvp_argnum(
    acp.array_from_args,
    lambda argnum,
    g,
    ans,
    args,
    kwargs: untake(g, argnum - 2, vspace(ans)),
)
defjvp(
    acp._array_from_scalar_or_array,
    None,
    None,
    lambda g,
    ans,
    args,
    kwargs,
    _: acp._array_from_scalar_or_array(args, kwargs, g),
)

# ----- Functions that are constant w.r.t. continuous inputs -----
# defjvp(acp.nan_to_num, lambda g, ans, x: acp.where(acp.isfinite(x), g, 0.))

# ----- Binary ufuncs (linear) -----
def_linear(acp.multiply)

# ----- Binary ufuncs -----
defjvp(
    acp.add,
    lambda g,
    ans,
    x,
    y: broadcast(g, ans),
    lambda g,
    ans,
    x,
    y: broadcast(g, ans),
)
defjvp(
    acp.subtract,
    lambda g,
    ans,
    x,
    y: broadcast(g, ans),
    lambda g,
    ans,
    x,
    y: broadcast(-g, ans),
)
defjvp(acp.divide, "same", lambda g, ans, x, y: -g * x / y ** 2)
defjvp(
    acp.maximum,
    lambda g,
    ans,
    x,
    y: g * balanced_eq(x, ans, y),
    lambda g,
    ans,
    x,
    y: g * balanced_eq(y, ans, x)
)
defjvp(
    acp.minimum,
    lambda g,
    ans,
    x,
    y: g * balanced_eq(x, ans, y),
    lambda g,
    ans,
    x,
    y: g * balanced_eq(y, ans, x)
)
defjvp(
    acp.fmax,
    lambda g,
    ans,
    x,
    y: g * balanced_eq(x, ans, y),
    lambda g,
    ans,
    x,
    y: g * balanced_eq(y, ans, x)
)
defjvp(
    acp.fmin,
    lambda g,
    ans,
    x,
    y: g * balanced_eq(x, ans, y),
    lambda g,
    ans,
    x,
    y: g * balanced_eq(y, ans, x)
)
defjvp(
    acp.logaddexp,
    lambda g,
    ans,
    x,
    y: g * acp.exp(x - ans),
    lambda g,
    ans,
    x,
    y: g * acp.exp(y - ans)
)
defjvp(
    acp.logaddexp2,
    lambda g,
    ans,
    x,
    y: g * 2 ** (x - ans),
    lambda g,
    ans,
    x,
    y: g * 2 ** (y - ans)
)
defjvp(acp.true_divide, "same", lambda g, ans, x, y: -g * x / y ** 2)
defjvp(
    acp.mod,
    lambda g,
    ans,
    x,
    y: broadcast(g, ans),
    lambda g,
    ans,
    x,
    y: -g * acp.floor(x / y)
)
defjvp(
    acp.remainder,
    lambda g,
    ans,
    x,
    y: broadcast(g, ans),
    lambda g,
    ans,
    x,
    y: -g * acp.floor(x / y)
)
defjvp(
    acp.power,
    lambda g,
    ans,
    x,
    y: g * y * x ** acp.where(y, y - 1, 1.),
    lambda g,
    ans,
    x,
    y: g * acp.log(replace_zero(x, 1.)) * x ** y
)
defjvp(
    acp.arctan2,
    lambda g,
    ans,
    x,
    y: g * y / (x ** 2 + y ** 2),
    lambda g,
    ans,
    x,
    y: g * -x / (x ** 2 + y ** 2)
)

# ----- Simple grads (linear) -----
defjvp(acp.negative, "same")
defjvp(acp.rad2deg, "same")
defjvp(acp.degrees, "same")
defjvp(acp.deg2rad, "same")
defjvp(acp.radians, "same")
defjvp(acp.reshape, "same")
defjvp(acp.roll, "same")
defjvp(acp.array_split, "same")
defjvp(acp.split, "same")
defjvp(acp.vsplit, "same")
defjvp(acp.hsplit, "same")
defjvp(acp.dsplit, "same")
defjvp(acp.ravel, "same")
defjvp(acp.expand_dims, "same")
defjvp(acp.squeeze, "same")
defjvp(acp.diag, "same")
defjvp(acp.diagonal, "same")
defjvp(acp.make_diagonal, "same")
defjvp(acp.flipud, "same")
defjvp(acp.fliplr, "same")
defjvp(acp.rot90, "same")
defjvp(acp.trace, "same")
defjvp(acp.full, "same", argnums=(1,))
defjvp(acp.triu,          'same')
defjvp(acp.tril,          'same')
defjvp(acp.swapaxes, "same")
defjvp(acp.rollaxis, "same")
defjvp(acp.moveaxis,      'same')
# def_linear(acp.cross)

# ----- Simple grads -----
defjvp(
    acp.abs,
    lambda g,
    ans,
    x: acp.real(g * replace_zero(acp.conj(x), 0.)) / replace_zero(ans, 1.),
)
# defjvp(acp.fabs,        lambda g, ans, x : acp.sign(x) * g)  # fabs doesn't take complex 
# numbers.
defjvp(acp.absolute, lambda g, ans, x: acp.real(g * acp.conj(x)) / ans)
defjvp(acp.reciprocal, lambda g, ans, x: -g / x ** 2)
defjvp(acp.exp, lambda g, ans, x: ans * g)
defjvp(acp.exp2, lambda g, ans, x: ans * acp.log(2) * g)
defjvp(acp.expm1, lambda g, ans, x: (ans + 1) * g)
defjvp(acp.log, lambda g, ans, x: g / x)
defjvp(acp.log2, lambda g, ans, x: g / x / acp.log(2))
defjvp(acp.log10, lambda g, ans, x: g / x / acp.log(10))
defjvp(acp.log1p, lambda g, ans, x: g / (x + 1))
defjvp(acp.sin, lambda g, ans, x: g * acp.cos(x))
defjvp(acp.cos, lambda g, ans, x: -g * acp.sin(x))
defjvp(acp.tan, lambda g, ans, x: g / acp.cos(x) ** 2)
defjvp(acp.arcsin, lambda g, ans, x: g / acp.sqrt(1 - x ** 2))
defjvp(acp.arccos, lambda g, ans, x: -g / acp.sqrt(1 - x ** 2))
defjvp(acp.arctan, lambda g, ans, x: g / (1 + x ** 2))
defjvp(acp.sinh, lambda g, ans, x: g * acp.cosh(x))
defjvp(acp.cosh, lambda g, ans, x: g * acp.sinh(x))
defjvp(acp.tanh, lambda g, ans, x: g / acp.cosh(x) ** 2)
defjvp(acp.arcsinh, lambda g, ans, x: g / acp.sqrt(x ** 2 + 1))
defjvp(acp.arccosh, lambda g, ans, x: g / acp.sqrt(x ** 2 - 1))
defjvp(acp.arctanh, lambda g, ans, x: g / (1 - x ** 2))
defjvp(acp.square, lambda g, ans, x: g * 2 * x)
defjvp(acp.sqrt, lambda g, ans, x: g * 0.5 * x ** -0.5)
# defjvp(acp.sinc,        lambda g, ans, x : g * (acp.cos(acp.pi*x)*acp.pi*x - 
# acp.sin(acp.pi*x))/(acp.pi*x**2))
defjvp(
    acp.clip,
    lambda g,
    ans,
    x,
    a_min,
    a_max: g * acp.logical_and(ans != a_min, ans != a_max)
)

# defjvp(acp.real_if_close, lambda g, ans, x : match_complex(ans, g))
defjvp(acp.real, lambda g, ans, x: acp.real(g))
defjvp(acp.imag, lambda g, ans, x: match_complex(ans, -1j * g))
defjvp(acp.conj, lambda g, ans, x: acp.conj(g))
defjvp(
    acp.angle,
    lambda g,
    ans,
    x: match_complex(ans, g * acp.conj(x * 1j) / acp.abs(x) ** 2)
)
defjvp(
    acp.where,
    None,
    lambda g,
    ans,
    c,
    x=None,
    y=None: acp.where(c, g, acp.zeros(g.shape)),
    lambda g,
    ans,
    c,
    x=None,
    y=None: acp.where(c, acp.zeros(g.shape), g),
)

# ----- Trickier grads -----
defjvp(acp.kron, "same", "same")
# defjvp(acp.diff,      'same')
defjvp(acp.repeat, "same")
defjvp(acp.tile, "same")
defjvp(acp.transpose, "same")
defjvp(acp.sum,       'same')
defjvp(acp.mean, "same")
defjvp(acp.prod, lambda g, ans, x, axis=None, keepdims=False: ans * acp.sum(g / x, axis=axis, keepdims=keepdims))
defjvp(
    acp.linspace,
    lambda g,
    ans,
    start,
    stop, *args, **kwargs: acp.linspace(g, 0, *args, **kwargs),
    lambda g,
    ans,
    start,
    stop, *args, **kwargs: acp.linspace(0, g, *args, **kwargs)
)


def forward_grad_np_var(g, ans, x, axis=None, ddof=0, keepdims=False):
    if axis is None:
        num_reps = acp.size(g)
    elif isinstance(axis, int):
        num_reps = g.shape[axis]
    elif isinstance(axis, tuple):
        num_reps = acp.prod(acp.array(np.shape(g))[list(axis)])

    x_minus_mean = acp.conj(x - acp.mean(x, axis=axis, keepdims=True))
    return (
        2.0
        * acp.sum(acp.real(g * x_minus_mean), axis=axis, keepdims=keepdims)
        / (num_reps - ddof)
    )


defjvp(acp.var, forward_grad_np_var)


def forward_grad_np_std(g, ans, x, axis=None, ddof=0, keepdims=False):
    if axis is None:
        num_reps = acp.size(g)
    elif isinstance(axis, int):
        num_reps = g.shape[axis]
    elif isinstance(axis, tuple):
        num_reps = acp.prod(acp.array(g.shape)[list(axis)])

    if num_reps <= 1:
        return acp.zeros_like(ans)

    x_minus_mean = acp.conj(x - acp.mean(x, axis=axis, keepdims=True))
    return (
        acp.sum(acp.real(g * x_minus_mean), axis=axis, keepdims=keepdims)
        / ((num_reps - ddof) * ans)
    )


defjvp(acp.std, forward_grad_np_std)


def fwd_grad_chooser(g, ans, x, axis=None, keepdims=False):
    if acp.isscalar(x):
        return g

    if not keepdims:
        if isinstance(axis, int):
            ans = acp.expand_dims(ans, axis)
        elif isinstance(axis, tuple):
            for ax in sorted(axis):
                ans = acp.expand_dims(ans, ax)
    chosen_locations = x == ans
    return (
        acp.sum((g * chosen_locations), axis=axis, keepdims=keepdims)
        / acp.sum(chosen_locations, axis=axis, keepdims=keepdims)
    )


defjvp(acp.max, fwd_grad_chooser)
defjvp(acp.min, fwd_grad_chooser)
defjvp(acp.amax, fwd_grad_chooser)
defjvp(acp.amin, fwd_grad_chooser)

defjvp(acp.cumsum, "same")

def_linear(acp.inner)
def_linear(acp.matmul)
def_linear(acp.dot)
def_linear(acp.tensordot)
def_linear(acp.outer)

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
            result.append(acp.zeros_like(axis_args[i]))
    return acp.concatenate_args(axis_args[0], *result)


defjvp_argnum(acp.concatenate_args, fwd_grad_concatenate_args)


def fwd_grad_sort(g, ans, x, axis=-1, kind="quicksort", order=None):
    sort_perm = acp.argsort(x, axis, kind, order)
    return g[sort_perm]


defjvp(acp.sort, fwd_grad_sort)
defjvp(acp.msort, lambda g, ans, x: fwd_grad_sort(g, ans, x, axis=0))


def fwd_grad_partition(g, ans, x, kth, axis=-1, kind="introselect", order=None):
    partition_perm = acp.argpartition(x, kth, axis, kind, order)
    return g[partition_perm]


defjvp(acp.partition, fwd_grad_partition)


def atleast_jvpmaker(fun):

    def jvp(g, ans, *arys):
        if len(arys) > 1:
            raise NotImplementedError("Can't handle multiple arguments yet.")

        return fun(g)

    return jvp


defjvp(acp.atleast_1d, atleast_jvpmaker(acp.atleast_1d))
defjvp(acp.atleast_2d, atleast_jvpmaker(acp.atleast_2d))
defjvp(acp.atleast_3d, atleast_jvpmaker(acp.atleast_3d))

def_linear(acp.einsum)

# TODO(mattjj): can we call np.broadcast_to or a related function instead?


def broadcast(x, target):
    target_shape, target_ndim, target_dtype, target_iscomplex = acp.metadata(target)
    while acp.ndim(x) < target_ndim:
        x = acp.expand_dims(x, 0)
    for axis, size in enumerate(x.shape):
        if size == 1:
            x = acp.repeat(x, target_shape[axis], axis=axis)
    if target_iscomplex and not acp.iscomplexobj(x):
        x = x + 0j  # TODO(mattjj): this might promote the dtype
    return x
