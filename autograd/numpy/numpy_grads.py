from autograd.core import getval

import operator as op
from . import numpy_wrapper as anp
from .numpy_extra import take

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
anp.nonzero.defgrad_is_zero()
anp.searchsorted.defgrad_is_zero()
anp.sign.defgrad_is_zero()
anp.ndim.defgrad_is_zero()
anp.shape.defgrad_is_zero()
anp.floor_divide.defgrad_is_zero(argnums=(0, 1))
anp.nan_to_num.defgrad_is_zero()
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

# ----- Binary ufuncs -----

I = lambda x : x
anp.negative.defgrad(lambda ans, x: op.neg)
anp.add.defgrad(lambda ans, x, y : unbroadcast(ans, x, I))
anp.add.defgrad(lambda ans, x, y : unbroadcast(ans, y, I), argnum=1)
anp.multiply.defgrad(lambda ans, x, y : unbroadcast(ans, x, lambda g : y * g))
anp.multiply.defgrad(lambda ans, x, y : unbroadcast(ans, y, lambda g : x * g), argnum=1)
anp.subtract.defgrad(lambda ans, x, y : unbroadcast(ans, x, I))
anp.subtract.defgrad(lambda ans, x, y : unbroadcast(ans, y, op.neg), argnum=1)
anp.divide.defgrad(lambda ans, x, y : unbroadcast(ans, x, lambda g : g / y))
anp.divide.defgrad(lambda ans, x, y : unbroadcast(ans, y, lambda g : - g * x / y**2), argnum=1)
anp.power.defgrad(lambda ans, x, y : unbroadcast(ans, x, lambda g : g * y * x ** (y - 1)))
anp.power.defgrad(lambda ans, x, y : unbroadcast(ans, y, lambda g : g * anp.log(x) * x ** y), argnum=1)
anp.maximum.defgrad(lambda ans, x, y : unbroadcast(ans, x, lambda g : g * (x == ans)))
anp.maximum.defgrad(lambda ans, x, y : unbroadcast(ans, y, lambda g : g * (y == ans)), argnum=1)
anp.minimum.defgrad(lambda ans, x, y : unbroadcast(ans, x, lambda g : g * (x == ans)))
anp.minimum.defgrad(lambda ans, x, y : unbroadcast(ans, y, lambda g : g * (y == ans)), argnum=1)
anp.logaddexp.defgrad(lambda ans, x, y : unbroadcast(ans, x, lambda g : g * anp.exp(x-ans)))
anp.logaddexp.defgrad(lambda ans, x, y : unbroadcast(ans, y, lambda g : g * anp.exp(y-ans)), argnum=1)
anp.logaddexp2.defgrad(lambda ans, x, y : unbroadcast(ans, x, lambda g : g * 2**(x-ans)))
anp.logaddexp2.defgrad(lambda ans, x, y : unbroadcast(ans, y, lambda g : g * 2**(y-ans)), argnum=1)


# ----- Simple grads -----

anp.abs.defgrad(     lambda ans, x : lambda g : anp.sign(x) * g)
anp.fabs.defgrad(    lambda ans, x : lambda g : anp.sign(x) * g)
anp.absolute.defgrad(lambda ans, x : lambda g : anp.sign(x) * g)
anp.true_divide.defgrad(lambda ans, x, y : lambda g : g / y)
anp.true_divide.defgrad(lambda ans, x, y : lambda g : - g * x / y**2, argnum=1)
anp.reciprocal.defgrad( lambda ans, x    : lambda g : - g / x**2)
anp.mod.defgrad(      lambda ans, x, y : lambda g : g)
anp.fmod.defgrad(     lambda ans, x, y : lambda g : g)
anp.remainder.defgrad(lambda ans, x, y : lambda g : g)
anp.mod.defgrad(      lambda ans, x, y : lambda g : -g * anp.floor(x/y), argnum=1)
anp.fmod.defgrad(     lambda ans, x, y : lambda g : -g * anp.floor(x/y), argnum=1)
anp.remainder.defgrad(lambda ans, x, y : lambda g : -g * anp.floor(x/y), argnum=1)
anp.exp.defgrad(   lambda ans, x : lambda g : ans * g)
anp.exp2.defgrad(  lambda ans, x : lambda g : ans * g * anp.log(2))
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
anp.square.defgrad(lambda ans, x : lambda g : g * 2 * x)
anp.sqrt.defgrad(  lambda ans, x : lambda g : g * 0.5 * x**-0.5)
anp.sinc.defgrad(  lambda ans, x : lambda g : g * (anp.cos(anp.pi*x)*anp.pi*x - anp.sin(anp.pi*x))/(anp.pi*x**2))
anp.reshape.defgrad( lambda ans, x, shape, order=None : lambda g : anp.reshape(g, x.shape, order=order))
anp.roll.defgrad(    lambda ans, x, shift, axis=None  : lambda g : anp.roll(g, -shift, axis=axis))
anp.ravel.defgrad(   lambda ans, x, order=None    : lambda g : anp.reshape(g, x.shape, order=order))
anp.expand_dims.defgrad(lambda ans, x, axis : lambda g : anp.squeeze(g, axis))
anp.squeeze.defgrad(    lambda ans, x, axis : lambda g : anp.repeat(g, x.shape[axis], axis))
anp.repeat.defgrad(     lambda ans, x, shape, axis  : lambda g : anp.sum(g, axis, keepdims=True))
anp.split.defgrad(      lambda ans, x, idxs, axis=0 : lambda g : anp.concatenate(g, axis=axis))
anp.diag.defgrad(       lambda ans, x               : lambda g : anp.diag(g))
anp.flipud.defgrad(     lambda ans, x,              : lambda g : anp.flipud(g))
anp.fliplr.defgrad(     lambda ans, x,              : lambda g : anp.fliplr(g))
anp.rot90.defgrad(      lambda ans, x, k=1          : lambda g : anp.rot90(g, -k))
anp.trace.defgrad(      lambda ans, x               : lambda g : g * anp.eye(x.shape[0]))
anp.full.defgrad(     lambda ans, shape, fill_value : lambda g : anp.sum(g), argnum=1)
anp.triu.defgrad(       lambda ans, x, k=0          : lambda g : anp.triu(g, k=k))
anp.tril.defgrad(       lambda ans, x, k=0          : lambda g : anp.tril(g, k=k))
anp.clip.defgrad(       lambda ans, x, a_min, a_max : lambda g : g * anp.logical_and(ans != a_min, ans != a_max))

# ----- Trickier grads -----

isarray = lambda x : isinstance(getval(x), anp.ndarray)
def make_grad_transpose(ans, x, axes=None):
    if axes is not None:
        axes = anp.argsort(axes)
    return lambda g : anp.transpose(g, axes)
anp.transpose.defgrad(make_grad_transpose)

def repeat_to_match_shape(x, axis, keepdims):
    """Returns a function that repeats an array along axis to get a given shape.
       Also returns the number of repetitions of the array."""
    if not isarray(x):
        return I, 1
    shape = x.shape
    if axis is None:
        return lambda g : anp.full(shape, g), anp.prod(shape)
    else:
        if keepdims:
            return lambda g : anp.repeat(g, shape[axis], axis), shape[axis]
        else:
            return lambda g : anp.repeat(anp.expand_dims(g, axis),
                                         shape[axis], axis), shape[axis]

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

def make_grad_chooser(ans, x, axis=None, keepdims=None):
    """Builds gradient of functions that choose a single item, such as min or max."""
    repeater, _ = repeat_to_match_shape(x, axis, keepdims)
    argmax_locations = x == repeater(ans)
    return lambda g: repeater(g) * argmax_locations
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

def make_grad_dot(argnum, ans, A, B):
    if anp.ndim(A) == 0 or anp.ndim(B) == 0:
        axes = ([], [])
    else:
        axes = ([A.ndim - 1], [max(0, B.ndim - 2)])
    return make_grad_tensordot(argnum, ans, A, B, axes=axes)
anp.dot.gradmaker = make_grad_dot

def make_grad_tensordot(argnum, ans, A, B, axes=2):
    if type(axes) is int:
        axes = (range(anp.ndim(A))[-axes:],
                range(anp.ndim(B))[:axes])

    def gradfun(g):
        N_axes_summed = len(axes[0])
        if argnum == 0:
            X, Y = A, B
            X_axes_summed, Y_axes_summed = axes
            g_axes_from_Y = range(anp.ndim(g))[(anp.ndim(X) - N_axes_summed):]
        else:
            X, Y = B, A
            X_axes_summed, Y_axes_summed = axes[::-1]
            g_axes_from_Y = range(anp.ndim(g))[:(anp.ndim(Y) - N_axes_summed)]

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
anp.tensordot.gradmaker = make_grad_tensordot

anp.outer.defgrad(lambda ans, a, b : lambda g : anp.dot(g, b.T))
anp.outer.defgrad(lambda ans, a, b : lambda g : anp.dot(a.T, g), argnum=1)

def make_grad_concatenate_args(argnum, ans, axis, *args):
    start = sum([a.shape[axis] for a in args[:argnum-1]])
    idxs = [slice(None)] * ans.ndim
    idxs[axis] = slice(start, start + args[argnum-1].shape[axis])
    return lambda g : take(g, idxs)
anp.concatenate_args.gradmaker = make_grad_concatenate_args

# ----- Handle broadcasting -----

def unbroadcast(ans, x, fun):
    if isarray(x):
        shape = x.shape
        def new_fun(g):
            result = fun(g)
            while result.ndim > len(shape):
                result = anp.sum(result, axis=0)
            for axis, size in enumerate(shape):
                if size is 1:
                    result = anp.sum(result, axis=axis, keepdims=True)
            assert result.shape == shape
            return result
    elif isarray(ans):
        new_fun = lambda g : anp.sum(fun(g))
    else:
        return fun
    new_fun.__name__ = "unbroadcast_{0}".format(fun.__name__)
    return new_fun
