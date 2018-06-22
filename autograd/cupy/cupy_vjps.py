from __future__ import absolute_import
from future.utils import string_types
from functools import partial
import cupy as ocp
from ..util import func
from . import cupy_wrapper as acp
from .cupy_boxes import ArrayBox
from autograd.extend import (
    primitive, vspace, defvjp, defvjp_argnum, SparseObject, VJPNode,
    register_notrace
)

# ----- Non-differentiable functions -----

nograd_functions = [
    acp.floor,
    acp.ceil,
    # acp.round,
    acp.rint,
    # acp.around,
    acp.fix,
    acp.trunc,
    # acp.all,
    # acp.any,
    acp.argmax,
    acp.argmin,
    acp.argpartition,
    acp.argsort,
    # acp.argwhere,
    acp.nonzero,
    acp.flatnonzero,
    acp.count_nonzero,
    # acp.searchsorted,
    acp.sign,
    # acp.ndim,
    # acp.shape,
    acp.floor_divide,
    acp.logical_and,
    acp.logical_or,
    acp.logical_not,
    acp.logical_xor,
    acp.isfinite,
    acp.isinf,
    acp.isnan,
    # acp.isneginf,
    # acp.isposinf,
    # acp.allclose,
    # acp.isclose,
    # acp.array_equal,
    # acp.array_equiv,
    acp.greater,
    acp.greater_equal,
    acp.less,
    acp.less_equal,
    acp.equal,
    acp.not_equal,
    acp.iscomplexobj,
    acp.iscomplex,
    acp.size,
    acp.isscalar,
    # acp.isreal,
    acp.zeros_like,
    acp.ones_like,
    acp.result_type,
]

for fun in nograd_functions:
    register_notrace(VJPNode, fun)

# ----- Functions that are constant w.r.t. continuous inputs -----

# defvjp(acp.nan_to_num, lambda ans, x: lambda g: acp.where(acp.isfinite(x), g, 0.))  # noqa: E501

# ----- Binary ufuncs -----

defvjp(
    acp.add,
    lambda ans,
    x,
    y: unbroadcast_f(x, lambda g: g),
    lambda ans,
    x,
    y: unbroadcast_f(y, lambda g: g),
)
defvjp(
    acp.multiply,
    lambda ans,
    x,
    y: unbroadcast_f(x, lambda g: y * g),
    lambda ans,
    x,
    y: unbroadcast_f(y, lambda g: x * g),
)
defvjp(
    acp.subtract,
    lambda ans,
    x,
    y: unbroadcast_f(x, lambda g: g),
    lambda ans,
    x,
    y: unbroadcast_f(y, lambda g: -g),
)
defvjp(
    acp.divide,
    lambda ans,
    x,
    y: unbroadcast_f(x, lambda g: g / y),
    lambda ans,
    x,
    y: unbroadcast_f(y, lambda g: -g * x / y ** 2)
)
defvjp(
    acp.maximum,
    lambda ans,
    x,
    y: unbroadcast_f(x, lambda g: g * balanced_eq(x, ans, y)),
    lambda ans,
    x,
    y: unbroadcast_f(y, lambda g: g * balanced_eq(y, ans, x)),
)
defvjp(
    acp.minimum,
    lambda ans,
    x,
    y: unbroadcast_f(x, lambda g: g * balanced_eq(x, ans, y)),
    lambda ans,
    x,
    y: unbroadcast_f(y, lambda g: g * balanced_eq(y, ans, x)),
)
defvjp(
    acp.fmax,
    lambda ans,
    x,
    y: unbroadcast_f(x, lambda g: g * balanced_eq(x, ans, y)),
    lambda ans,
    x,
    y: unbroadcast_f(y, lambda g: g * balanced_eq(y, ans, x)),
)
defvjp(
    acp.fmin,
    lambda ans,
    x,
    y: unbroadcast_f(x, lambda g: g * balanced_eq(x, ans, y)),
    lambda ans,
    x,
    y: unbroadcast_f(y, lambda g: g * balanced_eq(y, ans, x)),
)
defvjp(
    acp.logaddexp,
    lambda ans,
    x,
    y: unbroadcast_f(x, lambda g: g * acp.exp(x - ans)),
    lambda ans,
    x,
    y: unbroadcast_f(y, lambda g: g * acp.exp(y - ans)),
)
defvjp(
    acp.logaddexp2,
    lambda ans,
    x,
    y: unbroadcast_f(x, lambda g: g * 2 ** (x - ans)),
    lambda ans,
    x,
    y: unbroadcast_f(y, lambda g: g * 2 ** (y - ans))
)
defvjp(
    acp.true_divide,
    lambda ans,
    x,
    y: unbroadcast_f(x, lambda g: g / y),
    lambda ans,
    x,
    y: unbroadcast_f(y, lambda g: -g * x / y ** 2)
)
defvjp(
    acp.mod,
    lambda ans,
    x,
    y: unbroadcast_f(x, lambda g: g),
    lambda ans,
    x,
    y: unbroadcast_f(y, lambda g: -g * acp.floor(x / y)),
)
defvjp(
    acp.remainder,
    lambda ans,
    x,
    y: unbroadcast_f(x, lambda g: g),
    lambda ans,
    x,
    y: unbroadcast_f(y, lambda g: -g * acp.floor(x / y)),
)
defvjp(
    acp.power,
    lambda ans, x, y: unbroadcast_f(x, lambda g: g * y * x ** acp.where(y, y - 1., 1.)),
    lambda ans, x, y: unbroadcast_f(y, lambda g: g * acp.log(replace_zero(x, 1.)) * x ** y)
)
defvjp(
    acp.arctan2,
    lambda ans,
    x,
    y: unbroadcast_f(x, lambda g: g * y / (x ** 2 + y ** 2)),
    lambda ans,
    x,
    y: unbroadcast_f(y, lambda g: g * -x / (x ** 2 + y ** 2))
)
defvjp(
    acp.hypot,
    lambda ans,
    x,
    y: unbroadcast_f(x, lambda g: g * x / ans),
    lambda ans,
    x,
    y: unbroadcast_f(y, lambda g: g * y / ans),
)

# ----- Simple grads -----

defvjp(acp.negative, lambda ans, x: lambda g: -g)
defvjp(
    acp.abs,
    lambda ans,
    x: lambda g: g * replace_zero(acp.conj(x), 0.) / replace_zero(ans, 1.)  # noqa: E501
)
# defvjp(acp.fabs,     lambda ans, x : lambda g: acp.sign(x) * g)  # fabs doesn't take complex numbers.  # noqa: E501
defvjp(acp.absolute, lambda ans, x: lambda g: g * acp.conj(x) / ans)
defvjp(acp.reciprocal, lambda ans, x: lambda g: -g / x ** 2)
defvjp(acp.exp, lambda ans, x: lambda g: ans * g)
defvjp(acp.exp2, lambda ans, x: lambda g: ans * acp.log(2) * g)
defvjp(acp.expm1, lambda ans, x: lambda g: (ans + 1) * g)
defvjp(acp.log, lambda ans, x: lambda g: g / x)
defvjp(acp.log2, lambda ans, x: lambda g: g / x / acp.log(2))
defvjp(acp.log10, lambda ans, x: lambda g: g / x / acp.log(10))
defvjp(acp.log1p, lambda ans, x: lambda g: g / (x + 1))
defvjp(acp.sin, lambda ans, x: lambda g: g * acp.cos(x))
defvjp(acp.cos, lambda ans, x: lambda g: -g * acp.sin(x))
defvjp(acp.tan, lambda ans, x: lambda g: g / acp.cos(x) ** 2)
defvjp(acp.arcsin, lambda ans, x: lambda g: g / acp.sqrt(1 - x ** 2))
defvjp(acp.arccos, lambda ans, x: lambda g: -g / acp.sqrt(1 - x ** 2))
defvjp(acp.arctan, lambda ans, x: lambda g: g / (1 + x ** 2))
defvjp(acp.sinh, lambda ans, x: lambda g: g * acp.cosh(x))
defvjp(acp.cosh, lambda ans, x: lambda g: g * acp.sinh(x))
defvjp(acp.tanh, lambda ans, x: lambda g: acp.divide(g, acp.power(acp.cosh(x), 2)))
defvjp(acp.arcsinh, lambda ans, x: lambda g: g / acp.sqrt(x ** 2 + 1))
defvjp(acp.arccosh, lambda ans, x: lambda g: g / acp.sqrt(x ** 2 - 1))
defvjp(acp.arctanh, lambda ans, x: lambda g: g / (1 - x ** 2))
defvjp(acp.rad2deg, lambda ans, x: lambda g: g / acp.pi * 180.0)
defvjp(acp.degrees, lambda ans, x: lambda g: g / acp.pi * 180.0)
defvjp(acp.deg2rad, lambda ans, x: lambda g: g * acp.pi / 180.0)
defvjp(acp.radians, lambda ans, x: lambda g: g * acp.pi / 180.0)
defvjp(acp.square, lambda ans, x: lambda g: g * 2 * x)
defvjp(acp.sqrt, lambda ans, x: lambda g: g * 0.5 * x ** -0.5)
# defvjp(acp.sinc,    lambda ans, x : lambda g: g * (acp.cos(acp.pi*x)*acp.pi*x - acp.sin(acp.pi*x))/(acp.pi*x**2))  # noqa: E501
defvjp(
    acp.reshape,
    lambda ans,
    x,
    shape: lambda g: acp.reshape(g, x.shape),
)  # noqa: E501)
defvjp(
    acp.roll, lambda ans, x, shift, axis=None: lambda g: acp.roll(g, -shift, axis=axis)  # noqa: E501
)  # noqa: E501
defvjp(
    acp.array_split,
    lambda ans,
    ary,
    idxs,
    axis=0: lambda g: acp.concatenate(g, axis=axis),
)  # noqa: E501
defvjp(
    acp.split, lambda ans, ary, idxs, axis=0: lambda g: acp.concatenate(g, axis=axis)  # noqa: E501
)  # noqa: E501
defvjp(acp.vsplit, lambda ans, ary, idxs: lambda g: acp.concatenate(g, axis=0))
defvjp(acp.hsplit, lambda ans, ary, idxs: lambda g: acp.concatenate(g, axis=1))
defvjp(acp.dsplit, lambda ans, ary, idxs: lambda g: acp.concatenate(g, axis=2))
defvjp(
    acp.ravel,
    lambda ans,
    x: lambda g: acp.reshape(g, x.shape),
)  # noqa: E501
defvjp(acp.expand_dims, lambda ans, x, axis: lambda g: acp.reshape(g, x.shape))  # noqa: E501
defvjp(acp.squeeze, lambda ans, x, axis=None: lambda g: acp.reshape(g, x.shape))  # noqa: E501
defvjp(acp.diag, lambda ans, x, k=0: lambda g: acp.diag(g, k))
defvjp(acp.flipud, lambda ans, x: lambda g: acp.flipud(g))
defvjp(acp.fliplr, lambda ans, x: lambda g: acp.fliplr(g))
defvjp(acp.rot90, lambda ans, x, k=1: lambda g: acp.rot90(g, -k))
defvjp(
    acp.trace,
    lambda ans,
    x,
    offset=0: lambda g: acp.einsum(
        "ij,...->ij...", acp.eye(x.shape[0], x.shape[1], k=offset), g
    ),
)  # noqa: E501
defvjp(
    acp.full,
    lambda ans,
    shape,
    fill_value,
    dtype=None: lambda g: acp.sum(g),
    argnums=(1,),
)
defvjp(acp.triu,    lambda ans, x, k=0         : lambda g: acp.triu(g, k=k))
defvjp(acp.tril,    lambda ans, x, k=0         : lambda g: acp.tril(g, k=k))
defvjp(acp.clip,    lambda ans, x, a_min, a_max:
                        lambda g: g * acp.logical_and(ans != a_min, ans != a_max)
       )
defvjp(acp.swapaxes, lambda ans, x, axis1, axis2:
                         lambda g: acp.swapaxes(g, axis2, axis1)
       )
# defvjp(acp.moveaxis, lambda ans, a, source, destination: lambda g:
#                     acp.moveaxis(g, destination, source))
defvjp(
    acp.rollaxis,
    lambda ans,
    a,
    axis,
    start=0: lambda g: acp.rollaxis(g, start - 1, axis) if start
    > axis else acp.rollaxis(g, start, axis + 1),
)
# defvjp(acp.real_if_close, lambda ans, x : lambda g: match_complex(x, g))
defvjp(acp.real, lambda ans, x: lambda g: match_complex(x, g))
defvjp(acp.imag, lambda ans, x: lambda g: match_complex(x, -1j * g))
defvjp(acp.conj, lambda ans, x: lambda g: acp.conj(g))
# defvjp(acp.conjugate, lambda ans, x: lambda g: acp.conj(g))
defvjp(
    acp.angle,
    lambda ans,
    x: lambda g: match_complex(x, g * acp.conj(x * 1j) / acp.abs(x) ** 2)
)
defvjp(
    acp.where,
    None,
    lambda ans,
    c,
    x=None,
    y=None: lambda g: acp.where(c, g, acp.zeros(g.shape)),
    lambda ans,
    c,
    x=None,
    y=None: lambda g: acp.where(c, acp.zeros(g.shape), g),
)
# defvjp(acp.cross, lambda ans, a, b, axisa=-1, axisb=-1, axisc=-1, axis=None : lambda g:  # noqa: E501
#                   acp.cross(b, g, axisb, axisc, axisa, axis),
#                   lambda ans, a, b, axisa=-1, axisb=-1, axisc=-1, axis=None : lambda g:  # noqa: E501
#                   acp.cross(g, a, axisc, axisa, axisb, axis))
defvjp(
    acp.linspace,
    lambda ans,
    start,
    stop,
    num: lambda g: acp.dot(acp.linspace(1.0, 0.0, num), g),
    lambda ans,
    start,
    stop,
    num: lambda g: acp.dot(acp.linspace(0.0, 1.0, num), g),
)

defvjp(
    acp._astype,
    lambda ans,
    A,
    dtype,
    order="K",
    casting="unsafe",
    subok=True,
    copy=True: lambda g: acp._astype(g, A.dtype),
)

# ----- Trickier grads -----


# grad_diff is irrelevant for CuPy because CuPy does not currently implement
# diff.
def grad_diff(ans, a, n=1, axis=-1):
    nd = acp.ndim(a)
    ans_shape = acp.shape(ans)
    sl1 = [slice(None)] * nd
    sl1[axis] = slice(None, 1)

    sl2 = [slice(None)] * nd
    sl2[axis] = slice(-1, None)

    def undiff(g):
        if g.shape[axis] > 0:
            return acp.concatenate(
                (-g[sl1], -acp.diff(g, axis=axis), g[sl2]), axis=axis
            )

        shape = list(ans_shape)
        shape[axis] = 1
        return acp.zeros(shape)

    def helper(g, n):
        if n == 0:
            return g

        return helper(undiff(g), n - 1)

    return lambda g: helper(g, n)


# Commented out because CuPy does not implement diff.
# defvjp(acp.diff, grad_diff)


def grad_repeat(ans, x, repeats, axis=None):
    shape = x.shape

    def vjp(g):
        if axis is None:  # If axis is none, np.repeat() repeats the flattened array.  # noqa: E501
            expanded = acp.reshape(g, (acp.prod(shape),) + (repeats,))
            return acp.reshape(acp.sum(expanded, axis=1, keepdims=False), shape)  # noqa: E501

        else:
            if shape[axis] == 1:  # For this common case, the logic is simple.
                return acp.sum(g, axis=axis, keepdims=True)

            else:
                expanded = acp.reshape(
                    g, shape[0:axis + 1] + (repeats,) + shape[axis + 1:]
                )
                return acp.sum(expanded, axis=axis + 1, keepdims=False)

    return vjp


defvjp(acp.repeat, grad_repeat)


def grad_tile(ans, x, reps):
    reps = [reps] if acp.isscalar(reps) else reps
    x_shape = x.shape

    def vjp(g):
        for axis, rep in enumerate(reps):
            g = sum(acp.split(g, rep, axis))
        return acp.reshape(g, x_shape)

    return vjp


defvjp(acp.tile, grad_tile)


def grad_kron(argnum, ans, orig_A, orig_B):
    # kron has different promotion rules than dot. the reshapes are necessary
    # if and only if (1) orig_B is 1D or (2) orig_A and/or orig_B are 0D
    orig_A_shape = acp.shape(orig_A)
    orig_B_shape = acp.shape(orig_B)

    def vjp(G):
        A, B = acp.atleast_2d(orig_A), acp.atleast_2d(orig_B)
        shape = list(A.shape + B.shape)
        n = acp.ndim(A)
        shape[n - 1], shape[n] = shape[n], shape[n - 1]
        reshaped_G = acp.swapaxes(acp.reshape(G, shape), n - 1, n)
        if argnum == 0:
            return acp.reshape(
                acp.tensordot(reshaped_G, B, axes=acp.ndim(B)), orig_A_shape
            )

        else:
            return acp.reshape(
                acp.tensordot(A, reshaped_G, axes=acp.ndim(A)), orig_B_shape
            )

    return vjp


defvjp(acp.kron, partial(grad_kron, 0), partial(grad_kron, 1))


def grad_transpose(ans, x, axes=None):
    if axes is not None:
        axes = acp.argsort(axes)
    return lambda g: acp.transpose(g, axes)


defvjp(acp.transpose, grad_transpose)


def repeat_to_match_shape(g, shape, dtype, axis, keepdims):
    """Returns the array g repeated along axis to fit vector space vs.
       Also returns the number of repetitions of the array."""
    if shape == ():
        return g, 1
    axis = list(axis) if isinstance(axis, tuple) else axis
    new_shape = ocp.array(shape)
    new_shape[axis] = 1
    new_shape = tuple([int(i) for i in new_shape])
    num_reps = ocp.prod(ocp.array(shape)[axis])
    return acp.reshape(g, new_shape) + ocp.zeros(shape, dtype=dtype), num_reps


def grad_cp_sum(ans, x, axis=None, keepdims=False, dtype=None):
    shape, dtype = x.shape, acp.result_type(x)
    return lambda g: repeat_to_match_shape(g, shape, dtype, axis, keepdims)[0]


defvjp(acp.sum, grad_cp_sum)


def grad_cp_mean(ans, x, axis=None, keepdims=False):
    shape, dtype = x.shape, acp.result_type(x)

    def vjp(g):
        g_repeated, num_reps = repeat_to_match_shape(g, shape, dtype, axis, keepdims)  # noqa: E501
        return g_repeated / num_reps

    return vjp


defvjp(acp.mean, grad_cp_mean)


def grad_cp_prod(ans, x, axis=None, keepdims=False):  # TODO: Support tuples of axes.  # noqa: E501
    shape, dtype = x.shape, acp.result_type(x)

    def vjp(g):
        g_repeated, _ = repeat_to_match_shape(g * ans, shape, dtype, axis, keepdims)  # noqa: E501
        return g_repeated / x

    return vjp


defvjp(acp.prod, grad_cp_prod)


def grad_cp_var(ans, x, axis=None, ddof=0, keepdims=False):
    shape, _, dtype, iscomplex = acp.metadata(x)

    def vjp(g):
        if iscomplex:
            g = g + 0j
        g_repeated, num_reps = repeat_to_match_shape(g, shape, dtype, axis, keepdims)  # noqa: E501
        x_minus_mean = acp.conj(x - acp.mean(x, axis=axis, keepdims=True))
        return 2.0 * g_repeated * x_minus_mean / (num_reps - ddof)

    return vjp


defvjp(acp.var, grad_cp_var)


def grad_cp_std(ans, x, axis=None, ddof=0, keepdims=False):
    shape, _, dtype, iscomplex = acp.metadata(x)

    def vjp(g):
        if iscomplex:
            g = g + 0j
        g_repeated, num_reps = repeat_to_match_shape(
            g, shape, dtype, axis, keepdims
        )  # Avoid division by zero.
        if num_reps <= 1:
            return g_repeated * 0.0

        else:
            g_repeated, num_reps = repeat_to_match_shape(
                g / ans, shape, dtype, axis, keepdims
            )
            x_minus_mean = acp.conj(x - acp.mean(x, axis=axis, keepdims=True))
            return g_repeated * x_minus_mean / (num_reps - ddof)

    return vjp


defvjp(acp.std, grad_cp_std)


def grad_chooser(ans, x, axis=None, keepdims=None):
    shape, dtype = x.shape, acp.result_type(x)

    def vjp(g):
        """
        Builds gradient of functions that choose a single item, such as min
        or max.
        """
        g_repeated, _ = repeat_to_match_shape(g, shape, dtype, axis, keepdims)
        argmax_locations = x == repeat_to_match_shape(ans, shape, dtype, axis, keepdims)[0]
        return g_repeated * argmax_locations / ocp.sum(
            argmax_locations, axis=axis, keepdims=True
        )

    return vjp


defvjp(acp.max, grad_chooser)
defvjp(acp.min, grad_chooser)
defvjp(acp.amax, grad_chooser)
defvjp(acp.amin, grad_chooser)


def reverse_axis(x, axis):
    x = x.swapaxes(axis, 0)
    x = x[::-1, ...]
    return x.swapaxes(0, axis)


def grad_np_cumsum(ans, x, axis=None):

    def vjp(g):
        if axis:
            return reverse_axis(acp.cumsum(reverse_axis(g, axis), axis), axis)

        else:
            return acp.reshape(acp.cumsum(g[::-1], axis)[::-1], x.shape)

    return vjp


defvjp(acp.cumsum, grad_np_cumsum)


def grad_inner(argnum, ans, A, B):
    A_ndim, B_ndim = acp.ndim(A), acp.ndim(B)
    if A_ndim == 0 or B_ndim == 0:
        axes = ([], [])
    else:
        axes = ([A_ndim - 1], [B_ndim - 1])
    if argnum == 0:
        return lambda G: tensordot_adjoint_0(B, G, axes, A_ndim, B_ndim)

    elif argnum == 1:
        return lambda G: tensordot_adjoint_1(A, G, axes, A_ndim, B_ndim)


defvjp(acp.inner, partial(grad_inner, 0), partial(grad_inner, 1))


def grad_matmul(argnum, ans, A, B):
    A_ndim, B_ndim = acp.ndim(A), acp.ndim(B)
    if A_ndim == 0 or B_ndim == 0:
        raise ValueError("Scalar operands are not allowed, use '*' instead")

    elif A_ndim == 1 or B_ndim == 1 or (A_ndim == 2 and B_ndim == 2):
        axes = ([A_ndim - 1], [max(0, B_ndim - 2)])
        if argnum == 0:
            return lambda G: tensordot_adjoint_0(B, G, axes, A_ndim, B_ndim)

        elif argnum == 1:
            return lambda G: tensordot_adjoint_1(A, G, axes, A_ndim, B_ndim)

    else:
        return grad_einsum(argnum + 1, ans, ("...ij,...jk->...ik", A, B), None)


defvjp(acp.matmul, partial(grad_matmul, 0), partial(grad_matmul, 1))


@primitive
def dot_adjoint_0(B, G, A_ndim, B_ndim):
    # The adjoint of the operator
    # A |--> np.dot(A, B)
    if B_ndim == 0 or B_ndim == 1 or A_ndim == 0:
        contract_num = max(0, B_ndim - (A_ndim != 0))
        return ocp.tensordot(G, B, contract_num)

    else:
        return ocp.tensordot(G, ocp.swapaxes(B, -1, -2), B_ndim - 1)


@primitive
def dot_adjoint_1(A, G, A_ndim, B_ndim):
    # The adjoint of the operator
    # B |--> np.dot(A, B)
    needs_transpose = B_ndim > 1 and A_ndim != 0
    swap = (lambda x: ocp.swapaxes(x, -1, -2)) if needs_transpose else (lambda x: x)  # noqa: E501
    if A_ndim == 0 or A_ndim == 1 or B_ndim == 0:
        contract_num = max(0, A_ndim - (B_ndim != 0))
        return swap(ocp.tensordot(G, A, contract_num))

    else:
        return swap(
            ocp.tensordot(
                G, A, [range(-A_ndim - B_ndim + 2, -B_ndim + 1), range(A_ndim - 1)]  # noqa: E501
            )
        )


def dot_vjp_0(ans, A, B):
    A_ndim, B_ndim = acp.ndim(A), acp.ndim(B)
    return lambda g: dot_adjoint_0(B, g, A_ndim, B_ndim)


def dot_vjp_1(ans, A, B):
    A_ndim, B_ndim = acp.ndim(A), acp.ndim(B)
    return lambda g: dot_adjoint_1(A, g, A_ndim, B_ndim)


defvjp(acp.dot, dot_vjp_0, dot_vjp_1)

defvjp(dot_adjoint_0,
       lambda ans, B, g, An, Bn:
           lambda A: dot_adjoint_1(A, g, An, Bn),
       lambda ans, B, g, An, Bn:
           lambda A: acp.dot(A, B),
       )

defvjp(dot_adjoint_1,
       lambda ans, A, g, An, Bn:
           lambda B: dot_adjoint_0(B, g, An, Bn),
       lambda ans, A, g, An, Bn:
           lambda B: acp.dot(A, B),
       )


@primitive
def tensordot_adjoint_0(B, G, axes, A_ndim, B_ndim):
    # The adjoint of the operator
    # A |--> np.tensordot(A, B, axes)
    if B_ndim == 0:
        return G * B

    G_axes = ocp.arange(ocp.ndim(G))
    if type(axes) is int:
        axes = max(axes, 0)
        B_axes = ocp.arange(B_ndim)
        return ocp.tensordot(G, B, [G_axes[A_ndim - axes:], B_axes[axes:]])

    elif type(axes[0]) is int:
        axes = [axes[0] % A_ndim, axes[1] % B_ndim]
        B_axes = ocp.arange(B_ndim)
        return ocp.tensordot(G, B, [G_axes[A_ndim - 1:], ocp.delete(B_axes, axes[1])])  # noqa: E501

    else:
        A_axes = ocp.arange(A_ndim)
        B_axes = ocp.arange(B_ndim)
        summed_axes = [ocp.asarray(axes[0]) % A_ndim, ocp.asarray(axes[1]) % B_ndim]  # noqa: E501
        other_axes = [
            ocp.delete(A_axes, summed_axes[0]), ocp.delete(B_axes, summed_axes[1])  # noqa: E501
        ]
        out = ocp.tensordot(G, B, [G_axes[len(other_axes[0]):], other_axes[1]])
        perm = ocp.argsort(
            ocp.concatenate(
                (other_axes[0], summed_axes[0][ocp.argsort(summed_axes[1])])
            )
        )
        return ocp.transpose(out, perm)


@primitive
def tensordot_adjoint_1(A, G, axes, A_ndim, B_ndim):
    # The adjoint of the operator
    # B |--> np.tensordot(A, B, axes)
    if A_ndim == 0:
        return G * A

    G_axes = ocp.arange(ocp.ndim(G))
    if type(axes) is int:
        axes = max(axes, 0)
        A_axes = ocp.arange(A_ndim)
        return ocp.tensordot(A, G, [A_axes[:A_ndim - axes], G_axes[:A_ndim - axes]])  # noqa: E501

    elif type(axes[0]) is int:
        axes = [axes[0] % A_ndim, axes[1] % B_ndim]
        A_axes = ocp.arange(A_ndim)
        return ocp.tensordot(A, G, [ocp.delete(A_axes, axes[0]), G_axes[:A_ndim - 1]])  # noqa: E501

    else:
        A_axes = ocp.arange(A_ndim)
        B_axes = ocp.arange(B_ndim)
        summed_axes = [ocp.asarray(axes[0]) % A_ndim, ocp.asarray(axes[1]) % B_ndim]  # noqa: E501
        other_axes = [
            ocp.delete(A_axes, summed_axes[0]), ocp.delete(B_axes, summed_axes[1])  # noqa: E501
        ]
        out = ocp.tensordot(A, G, [other_axes[0], G_axes[:len(other_axes[0])]])
        perm = ocp.argsort(
            ocp.concatenate(
                (summed_axes[1][ocp.argsort(summed_axes[0])], other_axes[1])
            )
        )
        return ocp.transpose(out, perm)


def tensordot_vjp_0(ans, A, B, axes=2):
    A_ndim, B_ndim = acp.ndim(A), acp.ndim(B)
    return lambda G: tensordot_adjoint_0(B, G, axes, A_ndim, B_ndim)


def tensordot_vjp_1(ans, A, B, axes=2):
    A_ndim, B_ndim = acp.ndim(A), acp.ndim(B)
    return lambda G: tensordot_adjoint_1(A, G, axes, A_ndim, B_ndim)


defvjp(acp.tensordot, tensordot_vjp_0, tensordot_vjp_1)
defvjp(
    tensordot_adjoint_0,
    lambda ans,
    B,
    G,
    axes,
    An,
    Bn: lambda A: tensordot_adjoint_1(A, G, axes, An, Bn),
    lambda ans,
    B,
    G,
    axes,
    An,
    Bn: lambda A: acp.tensordot(A, B, axes),
)
defvjp(
    tensordot_adjoint_1,
    lambda ans,
    A,
    G,
    axes,
    An,
    Bn: lambda B: tensordot_adjoint_0(B, G, axes, An, Bn),
    lambda ans,
    A,
    G,
    axes,
    An,
    Bn: lambda B: acp.tensordot(A, B, axes),
)
defvjp(
    acp.outer,
    lambda ans,
    a,
    b: lambda g: acp.dot(g, b.T),
    lambda ans,
    a,
    b: lambda g: acp.dot(a.T, g),
)


def grad_concatenate_args(argnum, ans, axis_args, kwargs):
    axis, args = axis_args[0], axis_args[1:]
    sizes = [acp.shape(a)[axis] for a in args[:argnum]]
    start = sum(sizes[:-1])
    idxs = [slice(None)] * ans.ndim
    idxs[axis] = slice(start, start + sizes[-1])
    return lambda g: g[idxs]


defvjp_argnum(acp.concatenate_args, grad_concatenate_args)


def wrapped_reshape(x, *args, **kwargs):
    # The reshape method can be called like A.reshape((5,4)) or A.reshape(5,4).
    # The reshape function doesn't support both ways, so we have to wrap it.
    if isinstance(args[0], int):
        return acp.reshape(x, args, **kwargs)

    else:
        return acp.reshape(x, *args, **kwargs)


setattr(ArrayBox, "reshape", wrapped_reshape)


def grad_sort(ans, x, axis=-1, kind="quicksort", order=None):
    # TODO: Cast input with np.asanyarray()
    if len(x.shape) > 1:
        raise NotImplementedError(
            "Gradient of sort not implemented for multi-dimensional arrays."
        )

    sort_perm = acp.argsort(x, axis, kind, order)
    return lambda g: unpermuter(g, sort_perm)


defvjp(acp.sort, grad_sort)
defvjp(acp.msort, grad_sort)  # Until multi-D is allowed, these are the same.


def grad_partition(ans, x, kth, axis=-1, kind="introselect", order=None):
    # TODO: Cast input with np.asanyarray()
    if len(x.shape) > 1:
        raise NotImplementedError(
            "Gradient of partition not implemented for multi-dimensional arrays."  # noqa: E501
        )

    partition_perm = acp.argpartition(x, kth, axis, kind, order)
    return lambda g: unpermuter(g, partition_perm)


defvjp(acp.partition, grad_partition)


def unpermuter(g, permutation):
    unsort = acp.zeros(len(permutation), dtype=int)
    unsort[permutation] = list(range(len(permutation)))
    return g[unsort]


def grad_reshape_list(ans, *arys):
    if len(arys) > 1:
        raise NotImplementedError("Can't handle multiple arguments yet.")

    return lambda g: acp.reshape(g, acp.shape(arys[0]))


defvjp(acp.atleast_1d, grad_reshape_list)
defvjp(acp.atleast_2d, grad_reshape_list)
defvjp(acp.atleast_3d, grad_reshape_list)


def grad_einsum(argnum, ans, operands_, kwargs):
    result_meta = acp.metadata(operands_[argnum])

    def vjp(g):
        operands = operands_
        if isinstance(operands[0], string_types):  # using "ijk" convention.
            in_subs, out_subs, _ = acp.parse_einsum_input(*operands)
            strings, operands = operands[0], operands[1:]  # noqa: F841

            in_subs_list = in_subs.split(",")
            op_num = argnum - 1
            subs_wrt = in_subs_list[op_num]
            rest_of_ops = operands[:op_num] + operands[op_num + 1:]
            rest_of_subs = in_subs_list[:op_num] + in_subs_list[op_num + 1:]

            # subscripts that only appear in subs_wrt (and not in other
            # subscript lists or in the output) are implicitly being summed
            # out, as if contracted against a tensor of ones. we make that
            # tensor of ones explicit to handle the necessary vjp broadcasting
            # inside einsum.
            other_named_subs = set("".join([out_subs] + rest_of_subs))
            naked_summed = [
                (i, sub)
                for i, sub in enumerate(subs_wrt)
                if sub not in other_named_subs
            ]
            if naked_summed:
                naked_summed_dims, ones_subs = zip(*naked_summed)
                ones_subs = "".join(ones_subs)
                ones = ocp.ones(
                    ocp.array(operands[op_num].shape)[list(naked_summed_dims)]
                )
                new_input_subs = ",".join([out_subs, ones_subs] + rest_of_subs)
                new_operands = (g, ones) + rest_of_ops
            else:
                new_input_subs = ",".join([out_subs] + rest_of_subs)
                new_operands = (g,) + rest_of_ops

            new_subscripts = new_input_subs + "->" + subs_wrt
            return unbroadcast(acp.einsum(new_subscripts, *new_operands), result_meta)  # noqa: E501

        else:  # using (op0, sublist0, op1, sublist1, ..., sublistout) convention  # noqa: E501
            if len(operands) % 2 == 0:
                raise NotImplementedError("Need sublistout argument")

            operands = list(operands)
            rest_of_ops = [operands[-1]] + operands[:argnum] + operands[
                (argnum + 2):-1
            ] + [
                operands[argnum + 1]
            ]
            return unbroadcast_einsum(
                acp.einsum(g, *rest_of_ops), result_meta, operands[argnum + 1]
            )

    return vjp


defvjp_argnum(acp.einsum, grad_einsum)

defvjp(
    acp.diagonal,
    lambda ans,
    A,
    offset=0,
    axis1=0,
    axis2=1: lambda g: acp.make_diagonal(g, offset, axis1, axis2),
)
defvjp(
    acp.make_diagonal,
    lambda ans,
    D,
    offset=0,
    axis1=0,
    axis2=1: lambda g: acp.diagonal(g, offset, axis1, axis2),
)


def match_complex(target, x):
    target_iscomplex = acp.iscomplexobj(target)
    x_iscomplex = acp.iscomplexobj(x)
    if x_iscomplex and not target_iscomplex:
        return acp.real(x)

    elif not x_iscomplex and target_iscomplex:
        return x + 0j

    else:
        return x


def unbroadcast(x, target_meta, broadcast_idx=0):
    target_shape, target_ndim, dtype, target_iscomplex = target_meta
    while acp.ndim(x) > target_ndim:
        x = acp.sum(x, axis=broadcast_idx)
    for axis, size in enumerate(target_shape):
        if size == 1:
            x = acp.sum(x, axis=axis, keepdims=True)
    if acp.iscomplexobj(x) and not target_iscomplex:
        x = acp.real(x)
    return x


def unbroadcast_f(target, f):
    target_meta = acp.metadata(target)
    return lambda g: unbroadcast(f(g), target_meta)


def unbroadcast_einsum(x, target_meta, subscript):
    if Ellipsis not in subscript:
        return x

    elif subscript[0] == Ellipsis:
        return unbroadcast(x, target_meta, 0)

    elif subscript[-1] == Ellipsis:
        return unbroadcast(x, target_meta, -1)

    else:
        return unbroadcast(x, target_meta, subscript.index(Ellipsis))


def balanced_eq(x, z, y):
    return (x == z) / (1.0 + (x == y))


def replace_zero(x, val):
    return acp.where(x, x, val)


# ----- extra functions used internally  -----


def array_from_args_gradmaker(argnum, ans, args, kwargs):
    return lambda g: g[argnum - 2]


defvjp_argnum(acp.array_from_args, array_from_args_gradmaker)


def array_from_scalar_or_array_gradmaker(ans, array_args, array_kwargs, scarray):  # noqa: E501
    ndmin = array_kwargs.get("ndmin", 0)
    scarray_ndim = acp.ndim(scarray)
    if ndmin > scarray_ndim:
        return lambda g: acp.squeeze(g, axis=tuple(range(ndmin - scarray_ndim)))  # noqa: E501

    else:
        return lambda g: g


defvjp(
    acp._array_from_scalar_or_array,
    array_from_scalar_or_array_gradmaker,
    argnums=(2, 3),
)


@primitive
def untake(x, idx, vs):

    def mut_add(A):
        # in numpy codebase, this used to be:
        # onp.add.at(A, idx, x)
        # according to https://docs-cupy.chainer.org/en/stable/reference/ufunc.html?highlight=ufunc.at,
        # scatter_add is the correct function to use.
        # TODO: PR into cupy codebase the ability to use scatter_add with float64?
        ocp.scatter_add(A, idx, x)
        return A

    return SparseObject(vs, mut_add)


defvjp(
    func(ArrayBox.__getitem__), lambda ans, A, idx: lambda g: untake(g, idx, vspace(A))  # noqa: E501
)
defvjp(untake, lambda ans, x, idx, _: lambda g: g[idx])
