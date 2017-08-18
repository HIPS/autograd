from __future__ import absolute_import
from future.utils import string_types
import numpy as onp
from numpy.core.einsumfunc import _parse_einsum_input
from ..util import func
from autograd.tracer import primitive, getval
from autograd.vspace import vspace
from autograd.core import defvjp, defvjps, defvjp_is_zero, defvjp_argnum, SparseObject
from . import numpy_wrapper as anp
from .numpy_boxes import ArrayBox



# ----- Functions that are constant w.r.t. continuous inputs -----

defvjp_is_zero(anp.where, argnums=(0,))
defvjp(anp.nan_to_num, lambda ans, vs, gvs, x: lambda g: anp.where(anp.isfinite(x), g, 0.))

# ----- Binary ufuncs -----

defvjp(anp.add,         lambda ans, vs, gvs, x, y : lambda g: unbroadcast(vs, gvs, g))
defvjp(anp.add,         lambda ans, vs, gvs, x, y : lambda g: unbroadcast(vs, gvs, g), argnum=1)
defvjp(anp.multiply,    lambda ans, vs, gvs, x, y : lambda g: unbroadcast(vs, gvs, y * g))
defvjp(anp.multiply,    lambda ans, vs, gvs, x, y : lambda g: unbroadcast(vs, gvs, x * g), argnum=1)
defvjp(anp.subtract,    lambda ans, vs, gvs, x, y : lambda g: unbroadcast(vs, gvs, g))
defvjp(anp.subtract,    lambda ans, vs, gvs, x, y : lambda g: unbroadcast(vs, gvs, -g), argnum=1)
defvjp(anp.divide,      lambda ans, vs, gvs, x, y : lambda g: unbroadcast(vs, gvs,   g / y))
defvjp(anp.divide,      lambda ans, vs, gvs, x, y : lambda g: unbroadcast(vs, gvs, - g * x / y**2), argnum=1)
defvjp(anp.maximum,     lambda ans, vs, gvs, x, y : lambda g: unbroadcast(vs, gvs, g * balanced_eq(x, ans, y)))
defvjp(anp.maximum,     lambda ans, vs, gvs, x, y : lambda g: unbroadcast(vs, gvs, g * balanced_eq(y, ans, x)), argnum=1)
defvjp(anp.minimum,     lambda ans, vs, gvs, x, y : lambda g: unbroadcast(vs, gvs, g * balanced_eq(x, ans, y)))
defvjp(anp.minimum,     lambda ans, vs, gvs, x, y : lambda g: unbroadcast(vs, gvs, g * balanced_eq(y, ans, x)), argnum=1)
defvjp(anp.fmax,        lambda ans, vs, gvs, x, y : lambda g: unbroadcast(vs, gvs, g * balanced_eq(x, ans, y)))
defvjp(anp.fmax,        lambda ans, vs, gvs, x, y : lambda g: unbroadcast(vs, gvs, g * balanced_eq(y, ans, x)), argnum=1)
defvjp(anp.fmin,        lambda ans, vs, gvs, x, y : lambda g: unbroadcast(vs, gvs, g * balanced_eq(x, ans, y)))
defvjp(anp.fmin,        lambda ans, vs, gvs, x, y : lambda g: unbroadcast(vs, gvs, g * balanced_eq(y, ans, x)), argnum=1)
defvjp(anp.logaddexp,   lambda ans, vs, gvs, x, y : lambda g: unbroadcast(vs, gvs, g * anp.exp(x-ans)))
defvjp(anp.logaddexp,   lambda ans, vs, gvs, x, y : lambda g: unbroadcast(vs, gvs, g * anp.exp(y-ans)), argnum=1)
defvjp(anp.logaddexp2,  lambda ans, vs, gvs, x, y : lambda g: unbroadcast(vs, gvs, g * 2**(x-ans)))
defvjp(anp.logaddexp2,  lambda ans, vs, gvs, x, y : lambda g: unbroadcast(vs, gvs, g * 2**(y-ans)), argnum=1)
defvjp(anp.true_divide, lambda ans, vs, gvs, x, y : lambda g: unbroadcast(vs, gvs, g / y))
defvjp(anp.true_divide, lambda ans, vs, gvs, x, y : lambda g: unbroadcast(vs, gvs, - g * x / y**2), argnum=1)
defvjp(anp.mod,         lambda ans, vs, gvs, x, y : lambda g: unbroadcast(vs, gvs, g))
defvjp(anp.remainder,   lambda ans, vs, gvs, x, y : lambda g: unbroadcast(vs, gvs, g))
defvjp(anp.mod,         lambda ans, vs, gvs, x, y : lambda g: unbroadcast(vs, gvs, -g * anp.floor(x/y)), argnum=1)
defvjp(anp.remainder,   lambda ans, vs, gvs, x, y : lambda g: unbroadcast(vs, gvs, -g * anp.floor(x/y)), argnum=1)
defvjp(anp.power,
    lambda ans, vs, gvs, x, y : lambda g:
    unbroadcast(vs, gvs, g * y * x ** anp.where(y, y - 1, 1.)))
defvjp(anp.power,
    lambda ans, vs, gvs, x, y : lambda g:
    unbroadcast(vs, gvs, g * anp.log(replace_zero(x, 1.)) * x ** y), argnum=1)

# ----- Simple grads -----

defvjp(anp.negative, lambda ans, vs, gvs, x: lambda g: -g)
defvjp(anp.abs,
    lambda ans, vs, gvs, x : lambda g: g * replace_zero(anp.conj(x), 0.) / replace_zero(ans, 1.))
defvjp(anp.fabs,     lambda ans, vs, gvs, x : lambda g: anp.sign(x) * g)  # fabs doesn't take complex numbers.
defvjp(anp.absolute, lambda ans, vs, gvs, x : lambda g: g * anp.conj(x) / ans)
defvjp(anp.reciprocal, lambda ans, vs, gvs, x : lambda g: - g / x**2)
defvjp(anp.exp,    lambda ans, vs, gvs, x : lambda g: ans * g)
defvjp(anp.exp2,   lambda ans, vs, gvs, x : lambda g: ans * anp.log(2) * g)
defvjp(anp.expm1,  lambda ans, vs, gvs, x : lambda g: (ans + 1) * g)
defvjp(anp.log,    lambda ans, vs, gvs, x : lambda g: g / x)
defvjp(anp.log2,   lambda ans, vs, gvs, x : lambda g: g / x / anp.log(2))
defvjp(anp.log10,  lambda ans, vs, gvs, x : lambda g: g / x / anp.log(10))
defvjp(anp.log1p,  lambda ans, vs, gvs, x : lambda g: g / (x + 1))
defvjp(anp.sin,    lambda ans, vs, gvs, x : lambda g: g * anp.cos(x))
defvjp(anp.cos,    lambda ans, vs, gvs, x : lambda g: - g * anp.sin(x))
defvjp(anp.tan,    lambda ans, vs, gvs, x : lambda g: g / anp.cos(x) **2)
defvjp(anp.arcsin, lambda ans, vs, gvs, x : lambda g: g / anp.sqrt(1 - x**2))
defvjp(anp.arccos, lambda ans, vs, gvs, x : lambda g:-g / anp.sqrt(1 - x**2))
defvjp(anp.arctan, lambda ans, vs, gvs, x : lambda g: g / (1 + x**2))
defvjp(anp.sinh,   lambda ans, vs, gvs, x : lambda g: g * anp.cosh(x))
defvjp(anp.cosh,   lambda ans, vs, gvs, x : lambda g: g * anp.sinh(x))
defvjp(anp.tanh,   lambda ans, vs, gvs, x : lambda g: g / anp.cosh(x) **2)
defvjp(anp.arcsinh, lambda ans, vs, gvs, x : lambda g: g / anp.sqrt(x**2 + 1))
defvjp(anp.arccosh, lambda ans, vs, gvs, x : lambda g: g / anp.sqrt(x**2 - 1))
defvjp(anp.arctanh, lambda ans, vs, gvs, x : lambda g: g / (1 - x**2))
defvjp(anp.rad2deg, lambda ans, vs, gvs, x : lambda g: g / anp.pi * 180.0)
defvjp(anp.degrees, lambda ans, vs, gvs, x : lambda g: g / anp.pi * 180.0)
defvjp(anp.deg2rad, lambda ans, vs, gvs, x : lambda g: g * anp.pi / 180.0)
defvjp(anp.radians, lambda ans, vs, gvs, x : lambda g: g * anp.pi / 180.0)
defvjp(anp.square,  lambda ans, vs, gvs, x : lambda g: g * 2 * x)
defvjp(anp.sqrt,    lambda ans, vs, gvs, x : lambda g: g * 0.5 * x**-0.5)
defvjp(anp.sinc,    lambda ans, vs, gvs, x : lambda g: g * (anp.cos(anp.pi*x)*anp.pi*x - anp.sin(anp.pi*x))/(anp.pi*x**2))
defvjp(anp.reshape, lambda ans, vs, gvs, x, shape, order=None : lambda g: anp.reshape(g, vs.shape, order=order))
defvjp(anp.roll,    lambda ans, vs, gvs, x, shift, axis=None  : lambda g: anp.roll(g, -shift, axis=axis))
defvjp(anp.array_split, lambda ans, vs, gvs, ary, idxs, axis=0 : lambda g: anp.concatenate(g, axis=axis))
defvjp(anp.split,       lambda ans, vs, gvs, ary, idxs, axis=0 : lambda g: anp.concatenate(g, axis=axis))
defvjp(anp.vsplit,      lambda ans, vs, gvs, ary, idxs         : lambda g: anp.concatenate(g, axis=0))
defvjp(anp.hsplit,      lambda ans, vs, gvs, ary, idxs         : lambda g: anp.concatenate(g, axis=1))
defvjp(anp.dsplit,      lambda ans, vs, gvs, ary, idxs         : lambda g: anp.concatenate(g, axis=2))
defvjp(anp.ravel,   lambda ans, vs, gvs, x, order=None   : lambda g: anp.reshape(g, vs.shape, order=order))
defvjp(anp.expand_dims, lambda ans, vs, gvs, x, axis     : lambda g: anp.reshape(g, vs.shape))
defvjp(anp.squeeze, lambda ans, vs, gvs, x, axis=None    : lambda g: anp.reshape(g, vs.shape))
defvjp(anp.diag,    lambda ans, vs, gvs, x, k=0          : lambda g: anp.diag(g, k))
defvjp(anp.flipud,  lambda ans, vs, gvs, x,              : lambda g: anp.flipud(g))
defvjp(anp.fliplr,  lambda ans, vs, gvs, x,              : lambda g: anp.fliplr(g))
defvjp(anp.rot90,   lambda ans, vs, gvs, x, k=1          : lambda g: anp.rot90(g, -k))
defvjp(anp.trace,   lambda ans, vs, gvs, x, offset=0     : lambda g:
                    anp.einsum('ij,...->ij...', anp.eye(x.shape[0], x.shape[1], k=offset), g))
defvjp(anp.full,    lambda ans, vs, gvs, shape, fill_value, dtype=None : lambda g: anp.sum(g), argnum=1)
defvjp(anp.triu,    lambda ans, vs, gvs, x, k=0          : lambda g: anp.triu(g, k=k))
defvjp(anp.tril,    lambda ans, vs, gvs, x, k=0          : lambda g: anp.tril(g, k=k))
defvjp(anp.clip,    lambda ans, vs, gvs, x, a_min, a_max : lambda g: g * anp.logical_and(ans != a_min, ans != a_max))
defvjp(anp.swapaxes, lambda ans, vs, gvs, x, axis1, axis2: lambda g: anp.swapaxes(g, axis2, axis1))
defvjp(anp.moveaxis, lambda ans, vs, gvs, a, source, destination: lambda g:
                    anp.moveaxis(g, destination, source))
defvjp(anp.rollaxis, lambda ans, vs, gvs, a, axis, start=0: lambda g: anp.rollaxis(g, start - 1, axis) if start > axis
                                                 else anp.rollaxis(g, start, axis + 1))
defvjp(anp.real_if_close, lambda ans, vs, gvs, x : lambda g: match_complex(vs, g))
defvjp(anp.real,   lambda ans, vs, gvs, x   : lambda g: match_complex(vs, g))
defvjp(anp.imag,   lambda ans, vs, gvs, x   : lambda g: match_complex(vs, -1j * g))
defvjp(anp.conj,   lambda ans, vs, gvs, x   : lambda g: anp.conj(g))
defvjp(anp.conjugate, lambda ans, vs, gvs, x: lambda g: anp.conj(g))
defvjp(anp.angle,  lambda ans, vs, gvs, x   : lambda g: match_complex(vs, g * anp.conj(x * 1j) / anp.abs(x)**2))
defvjp(anp.where,  lambda ans, vs, gvs, c, x=None, y=None : lambda g: anp.where(c, g, anp.zeros(g.shape)), argnum=1)
defvjp(anp.where,  lambda ans, vs, gvs, c, x=None, y=None : lambda g: anp.where(c, anp.zeros(g.shape), g), argnum=2)
defvjp(anp.cross, lambda ans, vs, gvs, a, b, axisa=-1, axisb=-1, axisc=-1, axis=None : lambda g:
                  anp.cross(b, g, axisb, axisc, axisa, axis), argnum=0)
defvjp(anp.cross, lambda ans, vs, gvs, a, b, axisa=-1, axisb=-1, axisc=-1, axis=None : lambda g:
                  anp.cross(g, a, axisc, axisa, axisb, axis), argnum=1)
defvjp(anp.linspace, lambda ans, vs, gvs, start, stop, num : lambda g: anp.dot(anp.linspace(1.0, 0.0, num), g))
defvjp(anp.linspace, lambda ans, vs, gvs, start, stop, num : lambda g: anp.dot(anp.linspace(0.0, 1.0, num), g), argnum=1)

# ----- Trickier grads -----

def grad_diff(ans, vs, gvs, a, n=1, axis=-1):
    nd = len(vs.shape)
    sl1 = [slice(None)]*nd
    sl1[axis] = slice(None, 1)

    sl2 = [slice(None)]*nd
    sl2[axis] = slice(-1, None)

    def undiff(g):
        if g.shape[axis] > 0:
            return anp.concatenate((-g[sl1], -anp.diff(g, axis=axis), g[sl2]), axis=axis)
        shape = list(gvs.shape)
        shape[axis] = 1
        return anp.zeros(shape)

    def helper(g, n):
        if n == 0:
            return g
        return helper(undiff(g), n-1)
    return lambda g: helper(g, n)

defvjp(anp.diff, grad_diff)

def grad_repeat(ans, vs, gvs, x, repeats, axis=None):
    shape = vs.shape
    def vjp(g):
        if axis is None:  # If axis is none, np.repeat() repeats the flattened array.
            expanded = anp.reshape(g, (anp.prod(shape),) + (repeats,))
            return anp.reshape(anp.sum(expanded, axis=1, keepdims=False), shape)
        else:
            if shape[axis] == 1:  # For this common case, the logic is simple.
                return anp.sum(g, axis=axis, keepdims=True)
            else:
                expanded = anp.reshape(g, shape[0:axis+1] + (repeats,) + shape[axis+1:])
                return anp.sum(expanded, axis=axis+1, keepdims=False)
    return vjp

defvjp(anp.repeat, grad_repeat)

def grad_tile(ans, vs, gvs, x, reps):
    reps = [reps] if anp.isscalar(reps) else reps
    def vjp(g):
        for axis, rep in enumerate(reps):
            g = sum(anp.split(g, rep, axis))
        return anp.reshape(g, vs.shape)
    return vjp
defvjp(anp.tile, grad_tile)

def grad_kron(argnum, ans, vs, gvs, orig_A, orig_B):
    # kron has different promotion rules than dot. the reshapes are necessary if
    # and only if (1) orig_B is 1D or (2) orig_A and/or orig_B are 0D
    def vjp(G):
        A, B = anp.atleast_2d(orig_A), anp.atleast_2d(orig_B)
        shape = list(A.shape + B.shape)
        n = anp.ndim(A)
        shape[n-1], shape[n] = shape[n], shape[n-1]
        reshaped_G = anp.swapaxes(anp.reshape(G, shape), n-1, n)
        if argnum == 0:
            return anp.reshape(anp.tensordot(reshaped_G, B, axes=anp.ndim(B)), vs.shape)
        else:
            return anp.reshape(anp.tensordot(A, reshaped_G, axes=anp.ndim(A)), vs.shape)
    return vjp
defvjps(anp.kron, grad_kron, [0, 1])

def grad_transpose(ans, vs, gvs, x, axes=None):
    if axes is not None:
        axes = anp.argsort(axes)
    return lambda g: anp.transpose(g, axes)
defvjp(anp.transpose, grad_transpose)

def repeat_to_match_shape(g, vs, axis, keepdims):
    """Returns the array g repeated along axis to fit vector space vs.
       Also returns the number of repetitions of the array."""
    if vs.shape == ():
      return g, 1
    axis = list(axis) if isinstance(axis, tuple) else axis
    shape = onp.array(vs.shape)
    shape[axis] = 1
    num_reps = onp.prod(onp.array(vs.shape)[axis])
    return anp.reshape(g, shape) + vs.zeros(), num_reps

def grad_np_sum(ans, vs, gvs, x, axis=None, keepdims=False, dtype=None):
    return lambda g: repeat_to_match_shape(g, vs, axis, keepdims)[0]
defvjp(anp.sum, grad_np_sum)

def grad_np_mean(ans, vs, gvs, x, axis=None, keepdims=False):
    def vjp(g):
        g_repeated, num_reps = repeat_to_match_shape(g, vs, axis, keepdims)
        return g_repeated / num_reps
    return vjp
defvjp(anp.mean, grad_np_mean)

def grad_np_prod(ans, vs, gvs, x, axis=None, keepdims=False): # TODO: Support tuples of axes.
    def vjp(g):
        g_repeated, _ = repeat_to_match_shape(g * ans, vs, axis, keepdims)
        return g_repeated / x
    return vjp
defvjp(anp.prod, grad_np_prod)

def grad_np_var(ans, vs, gvs, x, axis=None, ddof=0, keepdims=False):
    def vjp(g):
        if vs.iscomplex:
            g = g + 0j
        g_repeated, num_reps = repeat_to_match_shape(g, vs, axis, keepdims)
        x_minus_mean = anp.conj(x - anp.mean(x, axis=axis, keepdims=True))
        return 2.0 * g_repeated * x_minus_mean / (num_reps - ddof)
    return vjp
defvjp(anp.var, grad_np_var)

def grad_np_std(ans, vs, gvs, x, axis=None, ddof=0, keepdims=False):
    def vjp(g):
        if vs.iscomplex:
            g = g + 0j
        g_repeated, num_reps = repeat_to_match_shape(g, vs, axis, keepdims)  # Avoid division by zero.
        if num_reps <= 1:
            return g_repeated * 0.0
        else:
            g_repeated, num_reps = repeat_to_match_shape(g / ans, vs, axis, keepdims)
            x_minus_mean = anp.conj(x - anp.mean(x, axis=axis, keepdims=True))
            return g_repeated * x_minus_mean / (num_reps - ddof)
    return vjp
defvjp(anp.std, grad_np_std)

def grad_chooser(ans, vs, gvs, x, axis=None, keepdims=None):
    def vjp(g):
        """Builds gradient of functions that choose a single item, such as min or max."""
        g_repeated, _ = repeat_to_match_shape(g, vs, axis, keepdims)
        argmax_locations = x == repeat_to_match_shape(ans, vs, axis, keepdims)[0]
        return g_repeated * argmax_locations \
            / onp.sum(argmax_locations, axis=axis, keepdims=True)
    return vjp
defvjp(anp.max, grad_chooser)
defvjp(anp.min, grad_chooser)
defvjp(anp.amax, grad_chooser)
defvjp(anp.amin, grad_chooser)

def reverse_axis(x, axis):
    x = x.swapaxes(axis, 0)
    x = x[::-1,...]
    return x.swapaxes(0, axis)

def grad_np_cumsum(ans, vs, gvs, x, axis=None):
    def vjp(g):
        if axis:
            return reverse_axis(anp.cumsum(reverse_axis(g, axis), axis), axis)
        else:
            return anp.reshape(anp.cumsum(g[::-1], axis)[::-1], x.shape)
    return vjp
defvjp(anp.cumsum, grad_np_cumsum)

def grad_inner(argnum, ans, vs, gvs, A, B):
    if anp.ndim(A) == 0 or anp.ndim(B) == 0:
        axes = ([], [])
    else:
        axes = ([A.ndim - 1], [B.ndim - 1])
    return grad_tensordot(argnum, ans, vs, gvs, A, B, axes=axes)
defvjps(anp.inner, grad_inner, [0, 1])

def grad_matmul(argnum, ans, vs, gvs, A, B):
    if anp.ndim(A) == 0 or anp.ndim(B) == 0:
        raise ValueError("Scalar operands are not allowed, use '*' instead")
    elif anp.ndim(A) == 1 or anp.ndim(B) == 1 or (anp.ndim(A) == 2 and anp.ndim(B) == 2):
        axes = ([A.ndim - 1], [max(0, B.ndim - 2)])
        return grad_tensordot(argnum, ans, vs, gvs, A, B, axes=axes)
    else:
        return grad_einsum(argnum + 1, ans, vs, gvs, ("...ij,...jk->...ik", A, B), None)
defvjps(anp.matmul, grad_matmul, [0, 1])

def grad_dot(argnum, ans, vs, gvs, A, B):
    # TODO: rewrite to allow garbage collection of A or B (depending on argnum)
    A_ndim, B_ndim = anp.ndim(A), anp.ndim(B)
    if A_ndim == 0 or B_ndim == 0:
        axes = ([], [])
    else:
        axes = ([A_ndim - 1], [max(0, B_ndim - 2)])
    return grad_tensordot(argnum, ans, vs, gvs, A, B, axes=axes)
defvjps(anp.dot, grad_dot, [0, 1])

def grad_tensordot(argnum, ans, vs, gvs, A, B, axes=2):
    def vjp(g):
        axes_ = axes
        if anp.size(A) == anp.size(B) == 0:
            return g * B if argnum == 0 else g * A

        A_ndim = anp.ndim(A)
        g_axes = onp.arange(anp.ndim(g))
        if type(axes_) is int:
            axes_ = max(axes_, 0)
            if argnum == 0:
                B_axes = onp.arange(anp.ndim(B))
                return anp.tensordot(g, B, [g_axes[A_ndim-axes_:], B_axes[axes_:]])
            else:
                A_axes = onp.arange(A_ndim)
                return anp.tensordot(A, g, [A_axes[:A_ndim-axes_], g_axes[:A_ndim-axes_]])
        elif type(axes_[0]) is int:
            B_ndim = anp.ndim(B)
            axes_ = [axes_[0] % A_ndim, axes_[1] % B_ndim]
            if argnum == 0:
                B_axes = onp.arange(B_ndim)
                return anp.tensordot(g, B, [g_axes[A_ndim-1:], onp.delete(B_axes, axes_[1])])
            else:
                A_axes = onp.arange(A_ndim)
                return anp.tensordot(A, g, [onp.delete(A_axes, axes_[0]), g_axes[:A_ndim-1]])
        else:
            B_ndim = anp.ndim(B)
            A_axes = onp.arange(A_ndim)
            B_axes = onp.arange(B_ndim)
            summed_axes = [onp.asarray(axes_[0]) % A_ndim,
                           onp.asarray(axes_[1]) % B_ndim]
            other_axes  = [onp.delete(A_axes, summed_axes[0]),
                           onp.delete(B_axes, summed_axes[1])]
            if argnum == 0:
                out = anp.tensordot(g, B, [g_axes[len(other_axes[0]):], other_axes[1]])
                perm = onp.argsort(onp.concatenate(
                    (other_axes[0], summed_axes[0][onp.argsort(summed_axes[1])])))
                return anp.transpose(out, perm)
            else:
                out = anp.tensordot(A, g, [other_axes[0], g_axes[:len(other_axes[0])]])
                perm = onp.argsort(onp.concatenate(
                    (summed_axes[1][onp.argsort(summed_axes[0])], other_axes[1])))
                return anp.transpose(out, perm)
    return vjp
defvjps(anp.tensordot, grad_tensordot, [0, 1])

defvjp(anp.outer, lambda ans, vs, gvs, a, b : lambda g: anp.dot(g, b.T))
defvjp(anp.outer, lambda ans, vs, gvs, a, b : lambda g: anp.dot(a.T, g), argnum=1)

def grad_concatenate_args(argnum, ans, vs, gvs, axis_args, kwargs):
    axis, args = axis_args[0], axis_args[1:]
    sizes = [a.shape[axis] for a in args[:argnum]]
    start = sum(sizes[:-1])
    idxs = [slice(None)] * ans.ndim
    idxs[axis] = slice(start, start + sizes[-1])
    return lambda g: g[idxs]
defvjp_argnum(anp.concatenate_args, grad_concatenate_args)

def wrapped_reshape(x, *args, **kwargs):
    # The reshape method can be called like A.reshape((5,4)) or A.reshape(5,4).
    # The reshape function doesn't support both ways, so we have to wrap it.
    if isinstance(args[0], int):
        return anp.reshape(x, args, **kwargs)
    else:
        return anp.reshape(x, *args, **kwargs)
setattr(ArrayBox, 'reshape', wrapped_reshape)

def grad_sort(ans, vs, gvs, x, axis=-1, kind='quicksort', order=None):
    #TODO: Cast input with np.asanyarray()
    if len(x.shape) > 1:
        raise NotImplementedError(
            "Gradient of sort not implemented for multi-dimensional arrays.")
    sort_perm = anp.argsort(x, axis, kind, order)
    return lambda g: unpermuter(g, sort_perm)
defvjp(anp.sort, grad_sort)
defvjp(anp.msort, grad_sort)  # Until multi-D is allowed, these are the same.

def grad_partition(ans, vs, gvs, x, kth, axis=-1, kind='introselect', order=None):
    #TODO: Cast input with np.asanyarray()
    if len(x.shape) > 1:
        raise NotImplementedError(
            "Gradient of partition not implemented for multi-dimensional arrays.")
    partition_perm = anp.argpartition(x, kth, axis, kind, order)
    return lambda g: unpermuter(g, partition_perm)
defvjp(anp.partition, grad_partition)

def unpermuter(g, permutation):
    unsort = anp.zeros(len(permutation), dtype=int)
    unsort[permutation] = list(range(len(permutation)))
    return g[unsort]

def grad_reshape_list(ans, vs, gvs, *arys):
    if len(arys) > 1:
        raise NotImplementedError("Can't handle multiple arguments yet.")
    return lambda g: anp.reshape(g, anp.shape(arys[0]))
defvjp(anp.atleast_1d, grad_reshape_list)
defvjp(anp.atleast_2d, grad_reshape_list)
defvjp(anp.atleast_3d, grad_reshape_list)

def grad_einsum(argnum, ans, vs, gvs, operands_, kwargs):
    def vjp(g):
        operands = operands_
        if isinstance(operands[0], string_types):  # using "ijk" convention.
            in_subs, out_subs, _ = _parse_einsum_input(tuple(map(getval, operands)))
            string, operands = operands[0], operands[1:]

            in_subs_list = in_subs.split(',')
            op_num = argnum - 1
            subs_wrt = in_subs_list[op_num]
            rest_of_ops = operands[:op_num] + operands[op_num+1:]
            rest_of_subs = in_subs_list[:op_num] + in_subs_list[op_num+1:]

            # subscripts that only appear in subs_wrt (and not in other subscript lists
            # or in the output) are implicitly being summed out, as if contracted
            # against a tensor of ones. we make that tensor of ones explicit to handle
            # the necessary vjp broadcasting inside einsum.
            other_named_subs = set(''.join([out_subs] + rest_of_subs))
            naked_summed = [(i, sub) for i, sub in enumerate(subs_wrt)
                            if sub not in other_named_subs]
            if naked_summed:
                naked_summed_dims, ones_subs = zip(*naked_summed)
                ones_subs = ''.join(ones_subs)
                ones = onp.ones(onp.array(operands[op_num].shape)[list(naked_summed_dims)])
                new_input_subs = ','.join([out_subs, ones_subs] + rest_of_subs)
                new_operands = (g, ones) + rest_of_ops
            else:
                new_input_subs = ','.join([out_subs] + rest_of_subs)
                new_operands = (g,) + rest_of_ops

            new_subscripts = new_input_subs + '->' + subs_wrt
            return unbroadcast(vs, gvs, anp.einsum(new_subscripts, *new_operands))
        else:  # using (op0, sublist0, op1, sublist1, ..., sublistout) convention
            if len(operands) % 2 == 0:
                raise NotImplementedError("Need sublistout argument")
            operands = list(operands)
            rest_of_ops = [operands[-1]] + operands[:argnum] + \
                    operands[(argnum+2):-1] + [operands[argnum+1]]
            return unbroadcast_einsum(vs, gvs, anp.einsum(g, *rest_of_ops), operands[argnum + 1])
    return vjp
defvjp_argnum(anp.einsum, grad_einsum)

defvjp(anp.diagonal,
    lambda ans, vs, gvs, A, offset=0, axis1=0, axis2=1 :
    lambda g: anp.make_diagonal(g, offset, axis1, axis2))
defvjp(anp.make_diagonal,
    lambda ans, vs, gvs, D, offset=0, axis1=0, axis2=1 :
    lambda g: anp.diagonal(g, offset, axis1, axis2))

def match_complex(vs, x):
    x_iscomplex = vspace(x).iscomplex
    if x_iscomplex and not vs.iscomplex:
        return anp.real(x)
    elif not x_iscomplex and vs.iscomplex:
        return x + 0j
    else:
        return x

def unbroadcast(vs, gvs, result, broadcast_idx=0):
    while anp.ndim(result) > len(vs.shape):
        result = anp.sum(result, axis=broadcast_idx)
    for axis, size in enumerate(vs.shape):
        if size == 1:
            result = anp.sum(result, axis=axis, keepdims=True)
    if gvs.iscomplex and not vs.iscomplex:
        result = anp.real(result)
    return result

def unbroadcast_einsum(vs, gvs, result, subscript):
    if Ellipsis not in subscript:
        return result
    elif subscript[0] == Ellipsis:
        return unbroadcast(vs, gvs, result, 0)
    elif subscript[-1] == Ellipsis:
        return unbroadcast(vs, gvs, result, -1)
    else:
        return unbroadcast(vs, gvs, result, subscript.index(Ellipsis))

def balanced_eq(x, z, y):
    return (x == z) / (1.0 + (x == y))

def replace_zero(x, val):
    return anp.where(x, x, val)

# ----- extra functions used internally  -----

def array_from_args_gradmaker(argnum, ans, vs, gvs, args, kwargs):
    return lambda g: g[argnum]
defvjp_argnum(anp.array_from_args, array_from_args_gradmaker)


@primitive
def untake(x, idx, vs):
    def mut_add(A):
        onp.add.at(A, idx, x)
        return A
    return SparseObject(vs, mut_add)
defvjp(func(ArrayBox.__getitem__), lambda ans, vs, gvs, A, idx: lambda g: untake(g, idx, vs))
defvjp(untake, lambda ans, vs, gvs, x, idx, _: lambda g: g[idx])
defvjp_is_zero(untake, argnums=(1, 2))
