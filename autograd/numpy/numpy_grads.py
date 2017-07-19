from __future__ import absolute_import
import numpy as onp
from numpy.core.einsumfunc import _parse_einsum_input

from autograd.core import primitive, getval, vspace
from . import numpy_wrapper as anp
from .numpy_extra import ArrayNode, take
from builtins import range, zip
from future.utils import string_types

# ----- Functions that are constant w.r.t. continuous inputs -----

anp.where.defvjp_is_zero(argnums=(0,))
anp.nan_to_num.defvjp(lambda g, ans, vs, gvs, x: anp.where(anp.isfinite(x), g, 0.))

# ----- Binary ufuncs -----

anp.add.defvjp(        lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, g))
anp.add.defvjp(        lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, g), argnum=1)
anp.multiply.defvjp(   lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, y * g))
anp.multiply.defvjp(   lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, x * g), argnum=1)
anp.subtract.defvjp(   lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, g))
anp.subtract.defvjp(   lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, -g), argnum=1)
anp.divide.defvjp(     lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs,   g / y))
anp.divide.defvjp(     lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, - g * x / y**2), argnum=1)
anp.maximum.defvjp(    lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, g * balanced_eq(x, ans, y)))
anp.maximum.defvjp(    lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, g * balanced_eq(y, ans, x)), argnum=1)
anp.minimum.defvjp(    lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, g * balanced_eq(x, ans, y)))
anp.minimum.defvjp(    lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, g * balanced_eq(y, ans, x)), argnum=1)
anp.fmax.defvjp(       lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, g * balanced_eq(x, ans, y)))
anp.fmax.defvjp(       lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, g * balanced_eq(y, ans, x)), argnum=1)
anp.fmin.defvjp(       lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, g * balanced_eq(x, ans, y)))
anp.fmin.defvjp(       lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, g * balanced_eq(y, ans, x)), argnum=1)
anp.logaddexp.defvjp(  lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, g * anp.exp(x-ans)))
anp.logaddexp.defvjp(  lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, g * anp.exp(y-ans)), argnum=1)
anp.logaddexp2.defvjp( lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, g * 2**(x-ans)))
anp.logaddexp2.defvjp( lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, g * 2**(y-ans)), argnum=1)
anp.true_divide.defvjp(lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, g / y))
anp.true_divide.defvjp(lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, - g * x / y**2), argnum=1)
anp.mod.defvjp(        lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, g))
anp.remainder.defvjp(  lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, g))
anp.mod.defvjp(        lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, -g * anp.floor(x/y)), argnum=1)
anp.remainder.defvjp(  lambda g, ans, vs, gvs, x, y : unbroadcast(vs, gvs, -g * anp.floor(x/y)), argnum=1)
anp.power.defvjp(
    lambda g, ans, vs, gvs, x, y :
    unbroadcast(vs, gvs, g * y * x ** anp.where(y, y - 1, 1.)))
anp.power.defvjp(
    lambda g, ans, vs, gvs, x, y :
    unbroadcast(vs, gvs, g * anp.log(replace_zero(x, 1.)) * x ** y), argnum=1)


# ----- Simple grads -----

anp.negative.defvjp(lambda g, ans, vs, gvs, x: -g)
anp.abs.defvjp(
    lambda g, ans, vs, gvs, x : g * replace_zero(anp.conj(x), 0.) / replace_zero(ans, 1.))
anp.fabs.defvjp(    lambda g, ans, vs, gvs, x : anp.sign(x) * g)  # fabs doesn't take complex numbers.
anp.absolute.defvjp(lambda g, ans, vs, gvs, x : g * anp.conj(x) / ans)
anp.reciprocal.defvjp(lambda g, ans, vs, gvs, x : - g / x**2)
anp.exp.defvjp(   lambda g, ans, vs, gvs, x : ans * g)
anp.exp2.defvjp(  lambda g, ans, vs, gvs, x : ans * anp.log(2) * g)
anp.expm1.defvjp( lambda g, ans, vs, gvs, x : (ans + 1) * g)
anp.log.defvjp(   lambda g, ans, vs, gvs, x : g / x)
anp.log2.defvjp(  lambda g, ans, vs, gvs, x : g / x / anp.log(2))
anp.log10.defvjp( lambda g, ans, vs, gvs, x : g / x / anp.log(10))
anp.log1p.defvjp( lambda g, ans, vs, gvs, x : g / (x + 1))
anp.sin.defvjp(   lambda g, ans, vs, gvs, x : g * anp.cos(x))
anp.cos.defvjp(   lambda g, ans, vs, gvs, x : - g * anp.sin(x))
anp.tan.defvjp(   lambda g, ans, vs, gvs, x : g / anp.cos(x) **2)
anp.arcsin.defvjp(lambda g, ans, vs, gvs, x : g / anp.sqrt(1 - x**2))
anp.arccos.defvjp(lambda g, ans, vs, gvs, x :-g / anp.sqrt(1 - x**2))
anp.arctan.defvjp(lambda g, ans, vs, gvs, x : g / (1 + x**2))
anp.sinh.defvjp(  lambda g, ans, vs, gvs, x : g * anp.cosh(x))
anp.cosh.defvjp(  lambda g, ans, vs, gvs, x : g * anp.sinh(x))
anp.tanh.defvjp(  lambda g, ans, vs, gvs, x : g / anp.cosh(x) **2)
anp.arcsinh.defvjp(lambda g, ans, vs, gvs, x : g / anp.sqrt(x**2 + 1))
anp.arccosh.defvjp(lambda g, ans, vs, gvs, x : g / anp.sqrt(x**2 - 1))
anp.arctanh.defvjp(lambda g, ans, vs, gvs, x : g / (1 - x**2))
anp.rad2deg.defvjp(lambda g, ans, vs, gvs, x : g / anp.pi * 180.0)
anp.degrees.defvjp(lambda g, ans, vs, gvs, x : g / anp.pi * 180.0)
anp.deg2rad.defvjp(lambda g, ans, vs, gvs, x : g * anp.pi / 180.0)
anp.radians.defvjp(lambda g, ans, vs, gvs, x : g * anp.pi / 180.0)
anp.square.defvjp( lambda g, ans, vs, gvs, x : g * 2 * x)
anp.sqrt.defvjp(   lambda g, ans, vs, gvs, x : g * 0.5 * x**-0.5)
anp.sinc.defvjp(   lambda g, ans, vs, gvs, x : g * (anp.cos(anp.pi*x)*anp.pi*x - anp.sin(anp.pi*x))/(anp.pi*x**2))
anp.reshape.defvjp(lambda g, ans, vs, gvs, x, shape, order=None : anp.reshape(g, vs.shape, order=order))
anp.roll.defvjp(   lambda g, ans, vs, gvs, x, shift, axis=None  : anp.roll(g, -shift, axis=axis))
anp.array_split.defvjp(lambda g, ans, vs, gvs, ary, idxs, axis=0 : anp.concatenate(g, axis=axis))
anp.split.defvjp(      lambda g, ans, vs, gvs, ary, idxs, axis=0 : anp.concatenate(g, axis=axis))
anp.vsplit.defvjp(     lambda g, ans, vs, gvs, ary, idxs         : anp.concatenate(g, axis=0))
anp.hsplit.defvjp(     lambda g, ans, vs, gvs, ary, idxs         : anp.concatenate(g, axis=1))
anp.dsplit.defvjp(     lambda g, ans, vs, gvs, ary, idxs         : anp.concatenate(g, axis=2))
anp.ravel.defvjp(  lambda g, ans, vs, gvs, x, order=None   : anp.reshape(g, vs.shape, order=order))
anp.expand_dims.defvjp(lambda g, ans, vs, gvs, x, axis     : anp.reshape(g, vs.shape))
anp.squeeze.defvjp(lambda g, ans, vs, gvs, x, axis=None    : anp.reshape(g, vs.shape))
anp.diag.defvjp(   lambda g, ans, vs, gvs, x, k=0          : anp.diag(g, k))
anp.flipud.defvjp( lambda g, ans, vs, gvs, x,              : anp.flipud(g))
anp.fliplr.defvjp( lambda g, ans, vs, gvs, x,              : anp.fliplr(g))
anp.rot90.defvjp(  lambda g, ans, vs, gvs, x, k=1          : anp.rot90(g, -k))
anp.trace.defvjp(  lambda g, ans, vs, gvs, x, offset=0     :
                    anp.einsum('ij,...->ij...', anp.eye(x.shape[0], x.shape[1], k=offset), g))
anp.full.defvjp(   lambda g, ans, vs, gvs, shape, fill_value, dtype=None : anp.sum(g), argnum=1)
anp.triu.defvjp(   lambda g, ans, vs, gvs, x, k=0          : anp.triu(g, k=k))
anp.tril.defvjp(   lambda g, ans, vs, gvs, x, k=0          : anp.tril(g, k=k))
anp.clip.defvjp(   lambda g, ans, vs, gvs, x, a_min, a_max : g * anp.logical_and(ans != a_min, ans != a_max))
anp.swapaxes.defvjp(lambda g, ans, vs, gvs, x, axis1, axis2: anp.swapaxes(g, axis2, axis1))
anp.moveaxis.defvjp(lambda g, ans, vs, gvs, a, source, destination:
                    anp.moveaxis(g, destination, source))
anp.rollaxis.defvjp(lambda g, ans, vs, gvs, a, axis, start=0: anp.rollaxis(g, start - 1, axis) if start > axis
                                                 else anp.rollaxis(g, start, axis + 1))
anp.real_if_close.defvjp(lambda g, ans, vs, gvs, x : match_complex(vs, g))
anp.real.defvjp(  lambda g, ans, vs, gvs, x   : match_complex(vs, g))
anp.imag.defvjp(  lambda g, ans, vs, gvs, x   : match_complex(vs, -1j * g))
anp.conj.defvjp(  lambda g, ans, vs, gvs, x   : anp.conj(g))
anp.conjugate.defvjp(lambda g, ans, vs, gvs, x: anp.conj(g))
anp.angle.defvjp( lambda g, ans, vs, gvs, x   : match_complex(vs, g * anp.conj(x * 1j) / anp.abs(x)**2))
anp.where.defvjp( lambda g, ans, vs, gvs, c, x=None, y=None : anp.where(c, g, anp.zeros(g.shape)), argnum=1)
anp.where.defvjp( lambda g, ans, vs, gvs, c, x=None, y=None : anp.where(c, anp.zeros(g.shape), g), argnum=2)
anp.cross.defvjp(lambda g, ans, vs, gvs, a, b, axisa=-1, axisb=-1, axisc=-1, axis=None :
                  anp.cross(b, g, axisb, axisc, axisa, axis), argnum=0)
anp.cross.defvjp(lambda g, ans, vs, gvs, a, b, axisa=-1, axisb=-1, axisc=-1, axis=None :
                  anp.cross(g, a, axisc, axisa, axisb, axis), argnum=1)
anp.linspace.defvjp(lambda g, ans, vs, gvs, start, stop, num : anp.dot(anp.linspace(1.0, 0.0, num), g))
anp.linspace.defvjp(lambda g, ans, vs, gvs, start, stop, num : anp.dot(anp.linspace(0.0, 1.0, num), g), argnum=1)

# ----- Trickier grads -----

def grad_diff(g, ans, vs, gvs, a, n=1, axis=-1):
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
    return helper(g, n)

anp.diff.defvjp(grad_diff)

def grad_repeat(g, ans, vs, gvs, x, repeats, axis=None):
    shape = x.shape
    if axis is None:  # If axis is none, np.repeat() repeats the flattened array.
        expanded = anp.reshape(g, (anp.prod(shape),) + (repeats,))
        return anp.reshape(anp.sum(expanded, axis=1, keepdims=False), shape)
    else:
        if shape[axis] == 1:  # For this common case, the logic is simple.
            return anp.sum(g, axis=axis, keepdims=True)
        else:
            expanded = anp.reshape(g, shape[0:axis+1] + (repeats,) + shape[axis+1:])
            return anp.sum(expanded, axis=axis+1, keepdims=False)
anp.repeat.defvjp(grad_repeat)

def grad_tile(g, ans, vs, gvs, x, reps):
    reps = [reps] if anp.isscalar(reps) else reps
    for axis, rep in enumerate(reps):
        g = sum(anp.split(g, rep, axis))
    return anp.reshape(g, x.shape)
anp.tile.defvjp(grad_tile)

def grad_kron(argnum, G, ans, vs, gvs, orig_A, orig_B):
    # kron has different promotion rules than dot. the reshapes are necessary if
    # and only if (1) orig_B is 1D or (2) orig_A and/or orig_B are 0D
    A, B = anp.atleast_2d(orig_A), anp.atleast_2d(orig_B)
    shape = list(A.shape + B.shape)
    n = anp.ndim(A)
    shape[n-1], shape[n] = shape[n], shape[n-1]
    reshaped_G = anp.swapaxes(anp.reshape(G, shape), n-1, n)
    if argnum == 0:
        return anp.reshape(anp.tensordot(reshaped_G, B, axes=anp.ndim(B)), vs.shape)
    else:
        return anp.reshape(anp.tensordot(A, reshaped_G, axes=anp.ndim(A)), vs.shape)
anp.kron.defvjps(grad_kron, [0, 1])

def grad_transpose(g, ans, vs, gvs, x, axes=None):
    if axes is not None:
        axes = anp.argsort(axes)
    return anp.transpose(g, axes)
anp.transpose.defvjp(grad_transpose)

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

def grad_np_sum(g, ans, vs, gvs, x, axis=None, keepdims=False, dtype=None):
    return repeat_to_match_shape(g, vs, axis, keepdims)[0]
anp.sum.defvjp(grad_np_sum)

def grad_np_mean(g, ans, vs, gvs, x, axis=None, keepdims=False):
    g_repeated, num_reps = repeat_to_match_shape(g, vs, axis, keepdims)
    return g_repeated / num_reps
anp.mean.defvjp(grad_np_mean)

def grad_np_prod(g, ans, vs, gvs, x, axis=None, keepdims=False): # TODO: Support tuples of axes.
    g_repeated, _ = repeat_to_match_shape(g * ans, vs, axis, keepdims)
    return g_repeated / x
anp.prod.defvjp(grad_np_prod)

def grad_np_var(g, ans, vs, gvs, x, axis=None, ddof=0, keepdims=False):
    if vs.iscomplex:
        g = g + 0j
    g_repeated, num_reps = repeat_to_match_shape(g, vs, axis, keepdims)
    x_minus_mean = anp.conj(x - anp.mean(x, axis=axis, keepdims=True))
    return 2.0 * g_repeated * x_minus_mean / (num_reps - ddof)
anp.var.defvjp(grad_np_var)

def grad_np_std(g, ans, vs, gvs, x, axis=None, ddof=0, keepdims=False):
    if vs.iscomplex:
        g = g + 0j
    g_repeated, num_reps = repeat_to_match_shape(g, vs, axis, keepdims)  # Avoid division by zero.
    if num_reps <= 1:
        return g_repeated * 0.0
    else:
        g_repeated, num_reps = repeat_to_match_shape(g / ans, vs, axis, keepdims)
        x_minus_mean = anp.conj(x - anp.mean(x, axis=axis, keepdims=True))
        return g_repeated * x_minus_mean / (num_reps - ddof)
anp.std.defvjp(grad_np_std)

def grad_chooser(g, ans, vs, gvs, x, axis=None, keepdims=None):
    """Builds gradient of functions that choose a single item, such as min or max."""
    g_repeated, _ = repeat_to_match_shape(g, vs, axis, keepdims)
    argmax_locations = x == repeat_to_match_shape(ans, vs, axis, keepdims)[0]
    return g_repeated * argmax_locations \
        / onp.sum(argmax_locations, axis=axis, keepdims=True)

anp.max.defvjp(grad_chooser)
anp.min.defvjp(grad_chooser)
anp.amax.defvjp(grad_chooser)
anp.amin.defvjp(grad_chooser)

def reverse_axis(x, axis):
    x = x.swapaxes(axis, 0)
    x = x[::-1,...]
    return x.swapaxes(0, axis)

def grad_np_cumsum(g, ans, vs, gvs, x, axis=None):
    if axis:
        return reverse_axis(anp.cumsum(reverse_axis(g, axis), axis), axis)
    else:
        return anp.reshape(anp.cumsum(g[::-1], axis)[::-1], x.shape)
anp.cumsum.defvjp(grad_np_cumsum)

def grad_inner(argnum, g, ans, vs, gvs, A, B):
    if anp.ndim(A) == 0 or anp.ndim(B) == 0:
        axes = ([], [])
    else:
        axes = ([A.ndim - 1], [B.ndim - 1])
    return grad_tensordot(argnum, g, ans, vs, gvs, A, B, axes=axes)
anp.inner.defvjps(grad_inner, [0, 1])

def grad_matmul(argnum, g, ans, vs, gvs, A, B):
    if anp.ndim(A) == 0 or anp.ndim(B) == 0:
        raise ValueError("Scalar operands are not allowed, use '*' instead")
    elif anp.ndim(A) == 1 or anp.ndim(B) == 1 or (anp.ndim(A) == 2 and anp.ndim(B) == 2):
        axes = ([A.ndim - 1], [max(0, B.ndim - 2)])
        return grad_tensordot(argnum, g, ans, vs, gvs, A, B, axes=axes)
    else:
        return grad_einsum(argnum + 1, g, ans, vs, gvs, ("...ij,...jk->...ik", A, B), None)
anp.matmul.defvjps(grad_matmul, [0, 1])

def grad_dot(argnum, g, ans, vs, gvs, A, B):
    A_ndim, B_ndim = anp.ndim(A), anp.ndim(B)
    if A_ndim == 0 or B_ndim == 0:
        axes = ([], [])
    else:
        axes = ([A_ndim - 1], [max(0, B_ndim - 2)])
    return grad_tensordot(argnum, g, ans, vs, gvs, A, B, axes=axes)
anp.dot.defvjps(grad_dot, [0, 1])

def grad_tensordot(argnum, g, ans, vs, gvs, A, B, axes=2):
    if anp.size(A) == anp.size(B) == 0:
        return g * B if argnum == 0 else g * A

    A_ndim = anp.ndim(A)
    g_axes = onp.arange(anp.ndim(g))
    if type(axes) is int:
        axes = max(axes, 0)
        if argnum == 0:
            B_axes = onp.arange(anp.ndim(B))
            return anp.tensordot(g, B, [g_axes[A_ndim-axes:], B_axes[axes:]])
        else:
            A_axes = onp.arange(A_ndim)
            return anp.tensordot(A, g, [A_axes[:A_ndim-axes], g_axes[:A_ndim-axes]])
    elif type(axes[0]) is int:
        B_ndim = anp.ndim(B)
        axes = [axes[0] % A_ndim, axes[1] % B_ndim]
        if argnum == 0:
            B_axes = onp.arange(B_ndim)
            return anp.tensordot(g, B, [g_axes[A_ndim-1:], onp.delete(B_axes, axes[1])])
        else:
            A_axes = onp.arange(A_ndim)
            return anp.tensordot(A, g, [onp.delete(A_axes, axes[0]), g_axes[:A_ndim-1]])
    else:
        B_ndim = anp.ndim(B)
        A_axes = onp.arange(A_ndim)
        B_axes = onp.arange(B_ndim)
        summed_axes = [onp.asarray(axes[0]) % A_ndim,
                       onp.asarray(axes[1]) % B_ndim]
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
anp.tensordot.defvjps(grad_tensordot, [0, 1])

anp.outer.defvjp(lambda g, ans, vs, gvs, a, b : anp.dot(g, b.T))
anp.outer.defvjp(lambda g, ans, vs, gvs, a, b : anp.dot(a.T, g), argnum=1)

def grad_concatenate_args(argnum, g, ans, vs, gvs, axis_args, kwargs):
    axis, args = axis_args[0], axis_args[1:]
    sizes = [a.shape[axis] for a in args[:argnum]]
    start = sum(sizes[:-1])
    idxs = [slice(None)] * ans.ndim
    idxs[axis] = slice(start, start + sizes[-1])
    return take(g, idxs)
anp.concatenate_args.vjp = grad_concatenate_args

def wrapped_reshape(x, *args, **kwargs):
    # The reshape method can be called like A.reshape((5,4)) or A.reshape(5,4).
    # The reshape function doesn't support both ways, so we have to wrap it.
    if isinstance(args[0], int):
        return anp.reshape(x, args, **kwargs)
    else:
        return anp.reshape(x, *args, **kwargs)
setattr(ArrayNode, 'reshape', wrapped_reshape)

def grad_sort(g, ans, vs, gvs, x, axis=-1, kind='quicksort', order=None):
    #TODO: Cast input with np.asanyarray()
    if len(x.shape) > 1:
        raise NotImplementedError(
            "Gradient of sort not implemented for multi-dimensional arrays.")
    sort_perm = anp.argsort(x, axis, kind, order)
    return unpermuter(g, sort_perm)
anp.sort.defvjp(grad_sort)
anp.msort.defvjp(grad_sort)  # Until multi-D is allowed, these are the same.

def grad_partition(g, ans, vs, gvs, x, kth, axis=-1, kind='introselect', order=None):
    #TODO: Cast input with np.asanyarray()
    if len(x.shape) > 1:
        raise NotImplementedError(
            "Gradient of partition not implemented for multi-dimensional arrays.")
    partition_perm = anp.argpartition(x, kth, axis, kind, order)
    return unpermuter(g, partition_perm)
anp.partition.defvjp(grad_partition)

def unpermuter(g, permutation):
    unsort = anp.zeros(len(permutation), dtype=int)
    unsort[permutation] = list(range(len(permutation)))
    return g[unsort]

def grad_reshape_list(g, ans, vs, gvs, *arys):
    if len(arys) > 1:
        raise NotImplementedError("Can't handle multiple arguments yet.")
    return anp.reshape(g, anp.shape(arys[0]))
anp.atleast_1d.defvjp(grad_reshape_list)
anp.atleast_2d.defvjp(grad_reshape_list)
anp.atleast_3d.defvjp(grad_reshape_list)

def grad_einsum(argnum, g, ans, vs, gvs, operands, kwargs):
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
anp.einsum.vjp = grad_einsum

@primitive
def make_diagonal(D, offset=0, axis1=0, axis2=1):
    # Numpy doesn't offer a complement to np.diagonal: a function to create new
    # diagonal arrays with extra dimensions. We need such a function for the
    # gradient of np.diagonal and it's also quite handy to have. So here it is.
    if not (offset==0 and axis1==-1 and axis2==-2):
        raise NotImplementedError("Currently make_diagonal only supports offset=0, axis1=-1, axis2=-2")

    # We use a trick: calling np.diagonal returns a view on the original array,
    # so we can modify it in-place. (only valid for numpy version >= 1.10.)
    new_array = onp.zeros(D.shape + (D.shape[-1],))
    new_array_diag = onp.diagonal(new_array, offset=0, axis1=-1, axis2=-2)
    new_array_diag.flags.writeable = True
    new_array_diag[:] = D
    return new_array

anp.make_diagonal = make_diagonal
anp.diagonal.defvjp(
    lambda g, ans, vs, gvs, A, offset=0, axis1=0, axis2=1 :
    anp.make_diagonal(g, offset, axis1, axis2))
anp.make_diagonal.defvjp(
    lambda g, ans, vs, gvs, D, offset=0, axis1=0, axis2=1 :
    anp.diagonal(g, offset, axis1, axis2))

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
