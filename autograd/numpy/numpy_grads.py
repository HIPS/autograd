from __future__ import absolute_import
import numpy as onp
import operator as op

from autograd.core import getval, primitive
from . import numpy_wrapper as anp
from .numpy_extra import ArrayNode, take
from builtins import range, zip
from future.utils import string_types

# ----- Functions that are constant w.r.t. continuous inputs -----

anp.floor.defgrad_is_zero()
anp.ceil.defgrad_is_zero()
anp.round.defgrad_is_zero()
anp.rint.defgrad_is_zero()
anp.around.defgrad_is_zero()
anp.fix.defgrad_is_zero()
anp.trunc.defgrad_is_zero()
anp.all.defgrad_is_zero()
anp.any.defgrad_is_zero()
anp.argmax.defgrad_is_zero()
anp.argmin.defgrad_is_zero()
anp.argpartition.defgrad_is_zero()
anp.argsort.defgrad_is_zero()
anp.argwhere.defgrad_is_zero()
anp.nonzero.defgrad_is_zero()
anp.flatnonzero.defgrad_is_zero()
anp.count_nonzero.defgrad_is_zero()
anp.searchsorted.defgrad_is_zero()
anp.sign.defgrad_is_zero()
anp.ndim.defgrad_is_zero()
anp.shape.defgrad_is_zero()
anp.floor_divide.defgrad_is_zero(argnums=(0, 1))
anp.logical_and.defgrad_is_zero(argnums=(0, 1))
anp.logical_or.defgrad_is_zero(argnums=(0, 1))
anp.logical_not.defgrad_is_zero(argnums=(0, 1))
anp.logical_xor.defgrad_is_zero(argnums=(0, 1))
anp.isfinite.defgrad_is_zero()
anp.isinf.defgrad_is_zero()
anp.isnan.defgrad_is_zero()
anp.isneginf.defgrad_is_zero()
anp.isposinf.defgrad_is_zero()
anp.allclose.defgrad_is_zero()
anp.isclose.defgrad_is_zero()
anp.array_equal.defgrad_is_zero()
anp.array_equiv.defgrad_is_zero()
anp.greater.defgrad_is_zero(argnums=(0, 1))
anp.greater_equal.defgrad_is_zero(argnums=(0, 1))
anp.less.defgrad_is_zero(argnums=(0, 1))
anp.less_equal.defgrad_is_zero(argnums=(0, 1))
anp.equal.defgrad_is_zero(argnums=(0, 1))
anp.not_equal.defgrad_is_zero(argnums=(0, 1))
anp.iscomplexobj.defgrad_is_zero()
anp.iscomplex.defgrad_is_zero()
anp.size.defgrad_is_zero()
anp.where.defgrad_is_zero(argnums=(0,))
anp.isscalar.defgrad_is_zero()
anp.isreal.defgrad_is_zero()
anp.zeros_like.defgrad_is_zero()
anp.ones_like.defgrad_is_zero()
anp.rollaxis.defgrad_is_zero(argnums=(1,2))
anp.copysign.defgrad_is_zero(argnums=(1,))
anp.nextafter.defgrad_is_zero(argnums=(1,))

# ----- Binary ufuncs -----

I = lambda x : x
anp.add.defgrad(lambda ans, x, y : unbroadcast(ans, x, I))
anp.add.defgrad(lambda ans, x, y : unbroadcast(ans, y, I), argnum=1)
anp.multiply.defgrad(lambda ans, x, y : unbroadcast(ans, x, lambda g : y * g))
anp.multiply.defgrad(lambda ans, x, y : unbroadcast(ans, y, lambda g : x * g), argnum=1)
anp.subtract.defgrad(lambda ans, x, y : unbroadcast(ans, x, I))
anp.subtract.defgrad(lambda ans, x, y : unbroadcast(ans, y, op.neg), argnum=1)
anp.divide.defgrad(lambda ans, x, y : unbroadcast(ans, x, lambda g :   g / y))
anp.divide.defgrad(lambda ans, x, y : unbroadcast(ans, y, lambda g : - g * x / y**2), argnum=1)
anp.power.defgrad(lambda ans, x, y : unbroadcast(ans, x, lambda g : g * y * x ** (anp.where(y, y - 1, 1.))))
anp.power.defgrad(
    lambda ans, x, y :
    unbroadcast(ans, y, lambda g : g * anp.log(anp.where(x, x, 1.)) * x ** y), argnum=1)
anp.maximum.defgrad(lambda ans, x, y : unbroadcast(ans, x, lambda g : g * balanced_eq(x, ans, y)))
anp.maximum.defgrad(lambda ans, x, y : unbroadcast(ans, y, lambda g : g * balanced_eq(y, ans, x)), argnum=1)
anp.minimum.defgrad(lambda ans, x, y : unbroadcast(ans, x, lambda g : g * balanced_eq(x, ans, y)))
anp.minimum.defgrad(lambda ans, x, y : unbroadcast(ans, y, lambda g : g * balanced_eq(y, ans, x)), argnum=1)
anp.fmax.defgrad(lambda ans, x, y : unbroadcast(ans, x, lambda g : g * balanced_eq(x, ans, y)))
anp.fmax.defgrad(lambda ans, x, y : unbroadcast(ans, y, lambda g : g * balanced_eq(y, ans, x)), argnum=1)
anp.fmin.defgrad(lambda ans, x, y : unbroadcast(ans, x, lambda g : g * balanced_eq(x, ans, y)))
anp.fmin.defgrad(lambda ans, x, y : unbroadcast(ans, y, lambda g : g * balanced_eq(y, ans, x)), argnum=1)
anp.logaddexp.defgrad(lambda ans, x, y : unbroadcast(ans, x, lambda g : g * anp.exp(x-ans)))
anp.logaddexp.defgrad(lambda ans, x, y : unbroadcast(ans, y, lambda g : g * anp.exp(y-ans)), argnum=1)
anp.logaddexp2.defgrad(lambda ans, x, y : unbroadcast(ans, x, lambda g : g * 2**(x-ans)))
anp.logaddexp2.defgrad(lambda ans, x, y : unbroadcast(ans, y, lambda g : g * 2**(y-ans)), argnum=1)
anp.true_divide.defgrad(lambda ans, x, y : unbroadcast(ans, x, lambda g : g / y))
anp.true_divide.defgrad(lambda ans, x, y : unbroadcast(ans, y, lambda g : - g * x / y**2), argnum=1)
anp.mod.defgrad(      lambda ans, x, y : unbroadcast(ans, x, I))
anp.remainder.defgrad(lambda ans, x, y : unbroadcast(ans, x, I))
anp.fmod.defgrad(     lambda ans, x, y : unbroadcast(ans, x, I))
anp.mod.defgrad(      lambda ans, x, y : unbroadcast(ans, y, lambda g : -g * anp.floor(x/y)), argnum=1)
anp.remainder.defgrad(lambda ans, x, y : unbroadcast(ans, y, lambda g : -g * anp.floor(x/y)), argnum=1)
anp.fmod.defgrad(     lambda ans, x, y : unbroadcast(ans, y, lambda g : -g * anp.fix(x/y)), argnum=1)
anp.hypot.defgrad(    lambda ans, x, y : unbroadcast(ans, x, lambda g : g * x / ans))
anp.hypot.defgrad(    lambda ans, x, y : unbroadcast(ans, y, lambda g : g * y / ans), argnum=1)
anp.arctan2.defgrad(  lambda ans, y, x : unbroadcast(ans, y, lambda g : g * x / (x ** 2 + y ** 2)))
anp.arctan2.defgrad(  lambda ans, y, x : unbroadcast(ans, x, lambda g : -g * y / (x ** 2 + y ** 2)), argnum=1)
anp.ldexp.defgrad(    lambda ans, x, y : unbroadcast(ans, x, lambda g : g * 2 ** y))
anp.copysign.defgrad( lambda ans, x, y : unbroadcast(ans, x, lambda g : g * anp.sign(x * y)))
anp.nextafter.defgrad(lambda ans, x, y : unbroadcast(ans, x, I))
anp.nan_to_num.defgrad(lambda ans, x: lambda g: anp.where(anp.isfinite(x), g, 0.))


# ----- Simple grads -----

anp.negative.defgrad(lambda ans, x: op.neg)
def abs_grad(ans, x):
    replace_zero = lambda x, val: anp.where(x, x, val)
    return lambda g: g * replace_zero(anp.conj(x), 0.) / replace_zero(ans, 1.)
anp.abs.defgrad(abs_grad)

# anp.abs.defgrad(     lambda ans, x : lambda g : g * anp.conj(x) / anp.where(ans, ans, 1.))
anp.fabs.defgrad(    lambda ans, x : lambda g : anp.sign(x) * g)  # fabs doesn't take complex numbers.
anp.absolute.defgrad(lambda ans, x : lambda g : g * anp.conj(x) / ans)
anp.reciprocal.defgrad(lambda ans, x : lambda g : - g / x**2)
anp.exp.defgrad(   lambda ans, x : lambda g : ans * g)
anp.exp2.defgrad(  lambda ans, x : lambda g : ans * anp.log(2) * g)
anp.expm1.defgrad( lambda ans, x : lambda g : (ans + 1) * g)
anp.log.defgrad(   lambda ans, x : lambda g : g / x)
anp.log2.defgrad(  lambda ans, x : lambda g : g / x / anp.log(2))
anp.log10.defgrad( lambda ans, x : lambda g : g / x / anp.log(10))
anp.log1p.defgrad( lambda ans, x : lambda g : g / (x + 1))
anp.sin.defgrad(   lambda ans, x : lambda g : g * anp.cos(x))
anp.cos.defgrad(   lambda ans, x : lambda g : - g * anp.sin(x))
anp.tan.defgrad(   lambda ans, x : lambda g : g / anp.cos(x) **2)
anp.arcsin.defgrad(lambda ans, x : lambda g : g / anp.sqrt(1 - x**2))
anp.arccos.defgrad(lambda ans, x : lambda g :-g / anp.sqrt(1 - x**2))
anp.arctan.defgrad(lambda ans, x : lambda g : g / (1 + x**2))
anp.sinh.defgrad(  lambda ans, x : lambda g : g * anp.cosh(x))
anp.cosh.defgrad(  lambda ans, x : lambda g : g * anp.sinh(x))
anp.tanh.defgrad(  lambda ans, x : lambda g : g / anp.cosh(x) **2)
anp.arcsinh.defgrad(lambda ans, x : lambda g : g / anp.sqrt(x**2 + 1))
anp.arccosh.defgrad(lambda ans, x : lambda g : g / anp.sqrt(x**2 - 1))
anp.arctanh.defgrad(lambda ans, x : lambda g : g / (1 - x**2))
anp.rad2deg.defgrad(lambda ans, x : lambda g : g / anp.pi * 180.0)
anp.degrees.defgrad(lambda ans, x : lambda g : g / anp.pi * 180.0)
anp.deg2rad.defgrad(lambda ans, x : lambda g : g * anp.pi / 180.0)
anp.radians.defgrad(lambda ans, x : lambda g : g * anp.pi / 180.0)
anp.square.defgrad( lambda ans, x : lambda g : g * 2 * x)
anp.sqrt.defgrad(   lambda ans, x : lambda g : g * 0.5 * x**-0.5)
anp.sinc.defgrad(   lambda ans, x : lambda g : g * (anp.cos(anp.pi*x)*anp.pi*x - anp.sin(anp.pi*x))/(anp.pi*x**2))
anp.reshape.defgrad(lambda ans, x, shape, order=None : lambda g : anp.reshape(g, anp.shape(x), order=order))
anp.roll.defgrad(   lambda ans, x, shift, axis=None  : lambda g : anp.roll(g, -shift, axis=axis))
anp.array_split.defgrad(lambda ans, ary, idxs, axis=0 : lambda g : anp.concatenate(g, axis=axis))
anp.split.defgrad(      lambda ans, ary, idxs, axis=0 : lambda g : anp.concatenate(g, axis=axis))
anp.vsplit.defgrad(     lambda ans, ary, idxs         : lambda g : anp.concatenate(g, axis=0))
anp.hsplit.defgrad(     lambda ans, ary, idxs         : lambda g : anp.concatenate(g, axis=1))
anp.dsplit.defgrad(     lambda ans, ary, idxs         : lambda g : anp.concatenate(g, axis=2))
anp.ravel.defgrad(  lambda ans, x, order=None   : lambda g : anp.reshape(g, anp.shape(x), order=order))
anp.expand_dims.defgrad(lambda ans, x, axis     : lambda g : anp.reshape(g, anp.shape(x)))
anp.squeeze.defgrad(lambda ans, x, axis=None    : lambda g : anp.reshape(g, anp.shape(x)))
anp.diag.defgrad(   lambda ans, x, k=0          : lambda g : anp.diag(g, k))
anp.flipud.defgrad( lambda ans, x,              : lambda g : anp.flipud(g))
anp.fliplr.defgrad( lambda ans, x,              : lambda g : anp.fliplr(g))
anp.rot90.defgrad(  lambda ans, x, k=1          : lambda g : anp.rot90(g, -k))
anp.trace.defgrad(  lambda ans, x, offset=0     : lambda g : anp.einsum('ij,...->ij...', anp.eye(x.shape[0], x.shape[1], k=offset), g))
anp.full.defgrad(   lambda ans, shape, fill_value, dtype=None : lambda g : anp.sum(g), argnum=1)
anp.triu.defgrad(   lambda ans, x, k=0          : lambda g : anp.triu(g, k=k))
anp.tril.defgrad(   lambda ans, x, k=0          : lambda g : anp.tril(g, k=k))
anp.clip.defgrad(   lambda ans, x, a_min, a_max : lambda g : g * anp.logical_and(ans != a_min, ans != a_max))
anp.swapaxes.defgrad(lambda ans, x, axis1, axis2: lambda g : anp.swapaxes(g, axis2, axis1))
anp.rollaxis.defgrad(lambda ans, a, axis, start=0: (lambda g : anp.rollaxis(g, start - 1, axis)) if start > axis
                                              else (lambda g : anp.rollaxis(g, start, axis + 1)))
anp.real_if_close.defgrad(lambda ans, x : lambda g : g)
anp.real.defgrad( lambda ans, x   : lambda g : g)
anp.imag.defgrad( lambda ans, x   : lambda g : -1j * g)
anp.conj.defgrad( lambda ans, x   : lambda g : anp.conj(g))
anp.angle.defgrad(lambda ans, x   : lambda g : g * anp.conj(x * 1j) / anp.abs(x)**2)
anp.where.defgrad(lambda ans, c, x=None, y=None : lambda g : anp.where(c, g, anp.zeros(g.shape)), argnum=1)
anp.where.defgrad(lambda ans, c, x=None, y=None : lambda g : anp.where(c, anp.zeros(g.shape), g), argnum=2)
anp.cross.defgrad(lambda ans, a, b, axisa=-1, axisb=-1, axisc=-1, axis=None:
                  lambda g : anp.cross(b, g, axisb, axisc, axisa, axis), argnum=0)
anp.cross.defgrad(lambda ans, a, b, axisa=-1, axisb=-1, axisc=-1, axis=None:
                  lambda g : anp.cross(g, a, axisc, axisa, axisb, axis), argnum=1)
anp.frexp.defgrad(lambda ans, x: lambda g : g[0] * 2.0 ** -ans[1])

# ----- Trickier grads -----

def make_grad_diff(ans, a, n=1, axis=-1):
    nd = len(a.shape)
    sl1 = [slice(None)]*nd
    sl1[axis] = slice(None, 1)

    sl2 = [slice(None)]*nd
    sl2[axis] = slice(-1, None)

    def undiff(g):
        if g.shape[axis] > 0:
            return anp.concatenate((-g[sl1], -anp.diff(g, axis=axis), g[sl2]), axis=axis)
        shape = list(g.shape)
        shape[axis] = 1
        return anp.zeros(shape)

    def gradfun(g):
        def helper(g, n):
            if n == 0:
                return g
            return helper(undiff(g), n-1)
        return helper(g, n)

    return gradfun
anp.diff.defgrad(make_grad_diff)

def make_grad_repeat(ans, x, repeats, axis=None):
    shape = x.shape
    if axis is None:  # If axis is none, np.repeat() repeats the flattened array.
        def grad_repeat(g):
            expanded = anp.reshape(g, (anp.prod(shape),) + (repeats,))
            return anp.reshape(anp.sum(expanded, axis=1, keepdims=False), shape)
        return grad_repeat
    else:
        if shape[axis] == 1:  # For this common case, the logic is simple.
            return lambda g: anp.sum(g, axis=axis, keepdims=True)
        else:
            def grad_repeat(g):
                expanded = anp.reshape(g, shape[0:axis+1] + (repeats,) + shape[axis+1:])
                return anp.sum(expanded, axis=axis+1, keepdims=False)
            return grad_repeat
anp.repeat.defgrad(make_grad_repeat)

def make_grad_tile(ans, x, reps):
    reps = [reps] if anp.isscalar(reps) else reps
    def tile_grad(g):
        for axis, rep in enumerate(reps):
            g = sum(anp.split(g, rep, axis))
        return anp.reshape(g, x.shape)
    return tile_grad
anp.tile.defgrad(make_grad_tile)

def make_grad_kron(argnum, ans, operands, kwargs):
    A = kwargs['a'] if 'a' in kwargs else operands[0]
    B = kwargs['b'] if 'b' in kwargs else operands[1]
    def blocks(G):
        return map(lambda blockrow: anp.split(blockrow, A.shape[1], 1),
                                    anp.split(G,        A.shape[0], 0))
    flat = lambda lst: [item for sublist in lst for item in sublist]

    if argnum == 0:
        Bflat = anp.ravel(B)
        return lambda G: \
            anp.array([[anp.dot(Bflat, anp.ravel(Gij)) for Gij in Gi]
                       for Gi in blocks(G)])
    else:
        Aflat = anp.ravel(A)
        return lambda G: \
            sum(aij * Gij for aij, Gij in zip(Aflat, flat(blocks(G))))
anp.kron.gradmaker = make_grad_kron

def make_grad_transpose(ans, x, axes=None):
    if axes is not None:
        axes = anp.argsort(axes)
    return lambda g : anp.transpose(g, axes)
anp.transpose.defgrad(make_grad_transpose)

isarray = lambda x : isinstance(getval(x), anp.ndarray)

def repeat_to_match_shape(x, axis, keepdims):
    """Returns a function that repeats an array along axis to get a given shape.
       Also returns the number of repetitions of the array."""
    assert isinstance(axis, (type(None), int, tuple))

    if not isarray(x):
        return I, 1
    shape = x.shape
    if axis is None:
        dtype=None
        if anp.iscomplexobj(x):
            dtype = getval(anp.array(x)).dtype   # np.full() has a bug for complex numbers
        if keepdims:
            return lambda g : anp.full(shape, anp.sum(g), dtype=dtype), anp.prod(shape)
        else:
            return lambda g : anp.full(shape, g, dtype=dtype), anp.prod(shape)
    elif isinstance(axis, int):
        if keepdims:
            return lambda g : anp.repeat(g, shape[axis], axis), shape[axis]
        else:
            return lambda g : anp.repeat(anp.expand_dims(g, axis),
                                         shape[axis], axis), shape[axis]
    else:
        repeats  = [shape[i] if i in axis else 1 for i in range(len(shape))]
        expanded = [shape[i] if i not in axis else 1 for i in range(len(shape))]
        num_reps = anp.prod(anp.array(shape)[list(axis)])

        if keepdims:
            return lambda g: anp.tile(g, repeats), num_reps
        else:
            return lambda g: anp.tile(anp.reshape(g, expanded), repeats), num_reps

def make_grad_np_sum(ans, x, axis=None, keepdims=False):
    repeater, _ = repeat_to_match_shape(x, axis, keepdims)
    return repeater
anp.sum.defgrad(make_grad_np_sum)

def make_grad_np_mean(ans, x, axis=None, keepdims=False):
    repeater, num_reps = repeat_to_match_shape(x, axis, keepdims)
    return lambda g: repeater(g) / num_reps
anp.mean.defgrad(make_grad_np_mean)

def make_grad_np_prod(ans, x, axis=None, keepdims=False): # TODO: Support tuples of axes.
    repeater, _ = repeat_to_match_shape(x, axis, keepdims)
    return lambda g: repeater(g * ans) / x
anp.prod.defgrad(make_grad_np_prod)

def make_grad_np_var(ans, x, axis=None, ddof=0, keepdims=False):
    repeater, num_reps = repeat_to_match_shape(x, axis, keepdims)
    x_minus_mean = anp.conj(x - anp.mean(x, axis=axis, keepdims=True))
    return lambda g: 2.0 * repeater(g) * x_minus_mean / (num_reps - ddof)
anp.var.defgrad(make_grad_np_var)

def make_grad_np_std(ans, x, axis=None, ddof=0, keepdims=False):
    repeater, num_reps = repeat_to_match_shape(x, axis, keepdims)
    if num_reps <= 1:
        return lambda g: repeater(0.0 * g)  # Avoid division by zero.
    else:
        x_minus_mean = anp.conj(x - anp.mean(x, axis=axis, keepdims=True))
        return lambda g: repeater(g / ans) * x_minus_mean / (num_reps - ddof)
anp.std.defgrad(make_grad_np_std)

def make_grad_chooser(ans, x, axis=None, keepdims=None):
    """Builds gradient of functions that choose a single item, such as min or max."""
    repeater, _ = repeat_to_match_shape(x, axis, keepdims)
    argmax_locations = x == repeater(ans)
    return lambda g: repeater(g) * argmax_locations \
        / onp.sum(argmax_locations, axis=axis, keepdims=True)
anp.max.defgrad(make_grad_chooser)
anp.min.defgrad(make_grad_chooser)
anp.amax.defgrad(make_grad_chooser)
anp.amin.defgrad(make_grad_chooser)

def reverse_axis(x, axis):
    x = x.swapaxes(axis, 0)
    x = x[::-1,...]
    return x.swapaxes(0, axis)

def make_grad_np_cumsum(ans, x, axis=None):
    if axis:
        return lambda g: reverse_axis(anp.cumsum(reverse_axis(g, axis)), axis)
    else:
        shape = x.shape
        return lambda g: anp.reshape(anp.cumsum(g[::-1], axis)[::-1], shape)
anp.cumsum.defgrad(make_grad_np_cumsum)

def make_grad_inner(argnum, ans, A, B):
    if anp.ndim(A) == 0 or anp.ndim(B) == 0:
        axes = ([], [])
    else:
        axes = ([A.ndim - 1], [B.ndim - 1])
    return make_grad_tensordot(argnum, ans, A, B, axes=axes)
anp.inner.defgrads(make_grad_inner, [0, 1])

def make_grad_dot(argnum, ans, A, B):
    if anp.ndim(A) == 0 or anp.ndim(B) == 0:
        axes = ([], [])
    else:
        axes = ([A.ndim - 1], [max(0, B.ndim - 2)])
    return make_grad_tensordot(argnum, ans, A, B, axes=axes)
anp.dot.defgrads(make_grad_dot, [0, 1])

def make_grad_matmul(argnum, ans, A, B):
    if anp.ndim(A) == 0 or anp.ndim(B) == 0:
        raise ValueError("Scalar operands are not allowed, use '*' instead")
    elif anp.ndim(A) == 1 or anp.ndim(B) == 1 or (anp.ndim(A) == 2 and anp.ndim(B) == 2):
        axes = ([A.ndim - 1], [max(0, B.ndim - 2)])
        return make_grad_tensordot(argnum, ans, A, B, axes=axes)
    else:
        return make_grad_einsum(argnum + 1, ans, ("...ij,...jk->...ik", A, B), None)
anp.matmul.defgrads(make_grad_matmul, [0, 1])

def make_grad_tensordot(argnum, ans, A, B, axes=2):
    if type(axes) is int:
        if axes > 0:
            axes = (list(range(anp.ndim(A)))[-axes:],
                    list(range(anp.ndim(B)))[:axes])
        else:
            axes = [(), ()] # summing over zero axes

        assert len(axes[0]) == len(axes[1])  # required by tensordot

    def convert_negative_indices(a, axes_list):
        axes = range(anp.ndim(a))
        return [axes[i] for i in axes_list]

    def gradfun(g):
        N_axes_summed = len(axes[0])
        if argnum == 0:
            X, Y = A, B
            X_axes_summed, Y_axes_summed = axes
            g_axes_from_Y = list(range(anp.ndim(g)))[(anp.ndim(X) - N_axes_summed):]
        else:
            X, Y = B, A
            X_axes_summed, Y_axes_summed = axes[::-1]
            g_axes_from_Y = list(range(anp.ndim(g)))[:(anp.ndim(Y) - N_axes_summed)]

        X_axes_summed, Y_axes_summed = map(
            convert_negative_indices, [X, Y], [X_axes_summed, Y_axes_summed])

        Y_axes_ignored = [i for i in range(anp.ndim(Y)) if i not in Y_axes_summed]
        result = anp.tensordot(g, Y, axes=[g_axes_from_Y, Y_axes_ignored])
        sorted_axes_pairs = sorted(zip(X_axes_summed, Y_axes_summed), key = lambda x : x[1])
        forward_permutation = ([i for i in range(anp.ndim(X)) if i not in X_axes_summed]
                             + [i for i, _ in sorted_axes_pairs])
        reverse_permutation = list(anp.argsort(forward_permutation))
        if result.ndim == 0:
            result = result[()]
        return anp.transpose(result, axes=reverse_permutation)
    return gradfun
anp.tensordot.defgrads(make_grad_tensordot, [0, 1])

anp.outer.defgrad(lambda ans, a, b : lambda g : anp.dot(g, b.T))
anp.outer.defgrad(lambda ans, a, b : lambda g : anp.dot(a.T, g), argnum=1)

def make_grad_concatenate_args(argnum, ans, axis_args, kwargs):
    axis, args = axis_args[0], axis_args[1:]
    start = sum([a.shape[axis] for a in args[:argnum-1]])
    idxs = [slice(None)] * ans.ndim
    idxs[axis] = slice(start, start + args[argnum-1].shape[axis])
    return lambda g : take(g, idxs)
anp.concatenate_args.gradmaker = make_grad_concatenate_args

def wrapped_reshape(x, *args, **kwargs):
    # The reshape method can be called like A.reshape((5,4)) or A.reshape(5,4).
    # The reshape function doesn't support both ways, so we have to wrap it.
    if isinstance(args[0], int):
        return anp.reshape(x, args, **kwargs)
    else:
        return anp.reshape(x, *args, **kwargs)
setattr(ArrayNode, 'reshape', wrapped_reshape)

def make_grad_sort(ans, x, axis=-1, kind='quicksort', order=None):
    #TODO: Cast input with np.asanyarray()
    if len(x.shape) > 1:
        raise NotImplementedError(
            "Gradient of sort not implemented for multi-dimensional arrays.")
    sort_perm = anp.argsort(x, axis, kind, order)
    return unpermuter(sort_perm)
anp.sort.defgrad(make_grad_sort)
anp.msort.defgrad(make_grad_sort)  # Until multi-D is allowed, these are the same.

def make_grad_partition(ans, x, kth, axis=-1, kind='introselect', order=None):
    #TODO: Cast input with np.asanyarray()
    if len(x.shape) > 1:
        raise NotImplementedError(
            "Gradient of partition not implemented for multi-dimensional arrays.")
    partition_perm = anp.argpartition(x, kth, axis, kind, order)
    return unpermuter(partition_perm)
anp.partition.defgrad(make_grad_partition)

def unpermuter(permutation):
    unsort = anp.zeros(len(permutation), dtype=int)
    unsort[permutation] = list(range(len(permutation)))
    return lambda g: g[unsort]

def make_grad_reshape_list(ans, *arys):
    if len(arys) > 1:
        raise NotImplementedError("Can't handle multiple arguments yet.")
    shape = anp.shape(arys[0])
    return lambda g: anp.reshape(g, shape)
anp.atleast_1d.defgrad(make_grad_reshape_list)
anp.atleast_2d.defgrad(make_grad_reshape_list)
anp.atleast_3d.defgrad(make_grad_reshape_list)

def make_grad_einsum(argnum, ans, operands, kwargs):
    # Gradient of einsum is obtained by swapping outgrad with the argument
    # being differentiated wrt.
    if isinstance(operands[0], string_types):  # using "ijk" convention.
        subscripts, operands = operands[0], operands[1:]
        if not '->' in subscripts:
            raise NotImplementedError("Need indices on both sides.")
        op_num = argnum - 1
        input_subs, output_subs = subscripts.split('->')
        input_subs_list = input_subs.split(',')
        subs_wrt = input_subs_list[op_num]
        rest_of_ops = operands[:op_num] + operands[op_num + 1:]
        rest_of_subs = input_subs_list[:op_num] + input_subs_list[op_num + 1:]
        new_subscripts = ','.join([output_subs] + rest_of_subs) + '->' + subs_wrt
        return unbroadcast_einsum(ans, operands[op_num],
                                  lambda g: anp.einsum(new_subscripts, *((g,) + rest_of_ops)),
                                  subs_wrt)
    else:  # Using (op0, sublist0, op1, sublist1..., sublistout) convention.
        if len(operands) % 2 == 0:
            raise NotImplementedError("Need sublistout argument")
        operands = list(operands)
        rest_of_ops = [operands[-1]] + operands[:argnum] + operands[(argnum+2):-1] + [operands[argnum+1]]
        return unbroadcast_einsum(ans, operands[argnum],
                                  lambda g: anp.einsum(g, *rest_of_ops),
                                  operands[argnum + 1])

anp.einsum.gradmaker = make_grad_einsum

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
anp.diagonal.defgrad(
    lambda ans, A, offset=0, axis1=0, axis2=1 :
    lambda g : anp.make_diagonal(g, offset, axis1, axis2))
anp.make_diagonal.defgrad(
    lambda ans, D, offset=0, axis1=0, axis2=1 :
    lambda g : anp.diagonal(g, offset, axis1, axis2))

# ----- Handle broadcasting -----

def unbroadcast(ans, x, gradfun, broadcast_idx=0):
    # x is the argument that we're differentiating with respect to.
    if isarray(x):
        shape = x.shape
        def new_fun(g):
            result = gradfun(g)
            while anp.ndim(result) > len(shape):
                result = anp.sum(result, axis=broadcast_idx)
            for axis, size in enumerate(shape):
                if size == 1:
                    result = anp.sum(result, axis=axis, keepdims=True)
            assert anp.shape(result) == shape
            return result
    elif isarray(ans):
        new_fun = lambda g : anp.sum(gradfun(g))
    else:
        return gradfun
    new_fun.__name__ = "unbroadcast_{0}".format(gradfun.__name__)
    return new_fun

def unbroadcast_einsum(ans, x, gradfun, subscript):
    if isinstance(subscript, string_types):
        if '...' not in subscript:
            return gradfun
        elif subscript.startswith('...'):
            return unbroadcast(ans, x, gradfun, 0)
        elif subscript.endswith('...'):
            return unbroadcast(ans, x, gradfun, -1)
        else:
            return unbroadcast(ans, x, gradfun, subscript.index('...'))
    else:
        if Ellipsis not in subscript:
            return gradfun
        elif subscript[0] == Ellipsis:
            return unbroadcast(ans, x, gradfun, 0)
        elif subscript[-1] == Ellipsis:
            return unbroadcast(ans, x, gradfun, -1)
        else:
            return unbroadcast(ans, x, gradfun, subscript.index(Ellipsis))

def balanced_eq(x, z, y):
    return (x == z) / (1.0 + (x == y))
