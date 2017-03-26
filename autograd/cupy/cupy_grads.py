from __future__ import absolute_import
import cupy as ocp
import numpy as np

from autograd.core import primitive, getval, vspace
from . import cupy_wrapper as acp
from .cupy_extra import CupyArrayNode, take, take_axis
from builtins import range, zip

acp.array.defvjp(lambda g, ans, vs, gvs, a, dtype=None, copy=True, ndmin=0:
                 acp.asnumpy(g))
acp.asnumpy.defvjp(lambda g, ans, vs, gvs, a, stream=None: acp.array(g))

acp.add.defvjp(     lambda g, ans, vs, gvs, x, y: unbroadcast(vs, gvs, g))
acp.add.defvjp(     lambda g, ans, vs, gvs, x, y: unbroadcast(vs, gvs, g), argnum=1)
acp.multiply.defvjp(lambda g, ans, vs, gvs, x, y: unbroadcast(vs, gvs, y * g))
acp.multiply.defvjp(lambda g, ans, vs, gvs, x, y: unbroadcast(vs, gvs, x * g), argnum=1)
acp.subtract.defvjp(lambda g, ans, vs, gvs, x, y: unbroadcast(vs, gvs, g))
acp.subtract.defvjp(lambda g, ans, vs, gvs, x, y: unbroadcast(vs, gvs, -g), argnum=1)
acp.divide.defvjp(  lambda g, ans, vs, gvs, x, y: unbroadcast(vs, gvs, g / y))
acp.divide.defvjp(  lambda g, ans, vs, gvs, x, y: unbroadcast(vs, gvs, - g * x / y**2), argnum=1)
acp.power.defvjp(lambda g, ans, vs, gvs, x, y:
                 unbroadcast(vs, gvs, g * y * x ** acp.where(y, y - 1, 1)))

acp.exp.defvjp(lambda g, ans, vs, gvs, x : ans * g)
acp.sinh.defvjp(lambda g, ans, vs, gvs, x: g * acp.cosh)
acp.cosh.defvjp(lambda g, ans, vs, gvs, x: g * acp.sinh(x))
acp.tanh.defvjp(lambda g, ans, vs, gvs, x: g / acp.cosh(x)**2)
acp.negative.defvjp(lambda g, ans, vs, gvs, x: -g)
acp.log.defvjp(lambda g, ans, vs, gvs, x : g / x)
acp.ravel.defvjp(lambda g, ans, vs, gvs, x: acp.reshape(g, vs.shape))
acp.reshape.defvjp(lambda g, ans, vs, gvs, x: acp.reshape(g, vs.shape))

def repeat_to_match_shape(g, vs, axis, keepdims):
    if vs.shape == ():
      return g, 1.
    axis = list(axis) if isinstance(axis, tuple) else axis
    shape = np.array(vs.shape)
    shape[axis] = 1.
    num_reps = np.prod(np.array(vs.shape)[axis])
    # NOTE(mattjj): cupy.reshape doesn't handle scalar arguments
    if not acp.isscalar(g):
        g = acp.reshape(g, shape)
    return g + vs.zeros(), num_reps

def grad_sum(g, ans, vs, gvs, x, axis=None, keepdims=False):
    return repeat_to_match_shape(g, vs, axis, keepdims)[0]
acp.sum.defvjp(grad_sum)

def grad_chooser(g, ans, vs, gvs, x, axis=None, keepdims=None):
    g_repeated, _ = repeat_to_match_shape(g, vs, axis, keepdims)
    argmax_locations = x == repeat_to_match_shape(ans, vs, axis, keepdims)[0]
    return g_repeated * argmax_locations \
        / ocp.sum(argmax_locations, axis=axis, keepdims=True)

acp.max.defvjp(grad_chooser)
acp.min.defvjp(grad_chooser)
acp.amax.defvjp(grad_chooser)
acp.amin.defvjp(grad_chooser)

def grad_mean(g, ans, vs, gvs, x, axis=None, keepdims=False):
    g_repeated, num_reps = repeat_to_match_shape(g, vs, axis, keepdims)
    return g_repeated / num_reps
acp.mean.defvjp(grad_mean)

def grad_dot(argnum, g, ans, vs, gvs, A, B):
    A_ndim, B_ndim = acp.ndim(A), acp.ndim(B)
    if A_ndim == 0 or B_ndim == 0:
        axes = ([], [])
    else:
        axes = ([A_ndim - 1], [max(0, B_ndim - 2)])
    return grad_tensordot(argnum, g, ans, vs, gvs, A, B, axes=axes)
acp.dot.defvjps(grad_dot, [0, 1])

def grad_tensordot(argnum, g, ans, vs, gvs, A, B, axes=2):
    A_ndim, B_ndim = acp.ndim(A), acp.ndim(B)
    g_ndim = len(gvs.shape)
    if type(axes) is int:
        if axes > 0:
            axes = (list(range(A_ndim))[-axes:],
                    list(range(B_ndim))[:axes])
        else:
            axes = [(), ()] # summing over zero axes

        assert len(axes[0]) == len(axes[1])  # required by tensordot

    def convert_negative_indices(a, axes_list):
        axes = range(acp.ndim(a))
        return [axes[i] for i in axes_list]

    N_axes_summed = len(axes[0])
    if argnum == 0:
        X, Y = A, B
        X_ndim, Y_ndim = A_ndim, B_ndim
        X_axes_summed, Y_axes_summed = axes
        g_axes_from_Y = list(range(g_ndim))[(X_ndim - N_axes_summed):]
    else:
        X, Y = B, A
        X_ndim, Y_ndim = B_ndim, A_ndim
        X_axes_summed, Y_axes_summed = axes[::-1]
        g_axes_from_Y = list(range(g_ndim))[:(Y_ndim - N_axes_summed)]

    X_axes_summed, Y_axes_summed = map(
        convert_negative_indices, [X, Y], [X_axes_summed, Y_axes_summed])

    Y_axes_ignored = [i for i in range(Y_ndim) if i not in Y_axes_summed]

    # NOTE(mattjj): cupy.tensordot doesn't handle scalar arguments because it checks
    # a.ndim and b.ndim instead of using an ndim function
    if g_ndim == 0:
        result = g * Y
    else:
        result = acp.tensordot(g, Y, axes=[g_axes_from_Y, Y_axes_ignored])
    sorted_axes_pairs = sorted(zip(X_axes_summed, Y_axes_summed), key =lambda x : x[1])
    forward_permutation = ([i for i in range(X_ndim) if i not in X_axes_summed]
                         + [i for i, _ in sorted_axes_pairs])
    reverse_permutation = list(np.argsort(forward_permutation))
    if result.ndim == 0:
        result = result[()]
    return acp.transpose(result, axes=reverse_permutation)
acp.tensordot.defvjps(grad_tensordot, [0, 1])

def grad_concatenate_args(argnum, g, ans, vs, gvs, axis_args, kwargs):
    axis, args = axis_args[0], axis_args[1:]
    start = sum([a.shape[axis] for a in args[:argnum-1]])
    return take_axis(g, np.arange(start, start + args[argnum-1].shape[axis]), axis)
acp.concatenate_args.vjp = grad_concatenate_args

def unbroadcast(vs, gvs, result, broadcast_idx=0):
    while acp.ndim(result) > len(vs.shape):
        result = acp.sum(result, axis=broadcast_idx)
    for axis, size in enumerate(vs.shape):
        if size == 1:
            result = acp.sum(result, axis=axis, keepdims=True)
    return result
