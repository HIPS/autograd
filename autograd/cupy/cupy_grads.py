from __future__ import absolute_import
import cupy as ocp
import numpy as np

from autograd.core import primitive, getval, vspace
from autograd.numpy.numpy_grads import balanced_eq
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
acp.maximum.defvjp( lambda g, ans, vs, gvs, x, y: unbroadcast(vs, gvs, g * balanced_eq(x, ans, y)))
acp.maximum.defvjp( lambda g, ans, vs, gvs, x, y: unbroadcast(vs, gvs, g * balanced_eq(y, ans, x)),
                   argnum=1)
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
    A_ndim = acp.ndim(A)
    g_axes = np.arange(acp.ndim(g))
    g_ndim = len(gvs.shape)

    if g_ndim == 0:
        if argnum == 0:
          return g * B
        else:
          return A * g
    if type(axes) is int:
        axes = max(axes, 0)
        if argnum == 0:
            B_axes = np.arange(acp.ndim(B))
            return acp.tensordot(g, B, [g_axes[A_ndim-axes:], B_axes[axes:]])
        else:
            A_axes = np.arange(A_ndim)
            return acp.tensordot(A, g, [A_axes[:A_ndim-axes], g_axes[:A_ndim-axes]])
    elif type(axes[0]) is int:
        B_ndim = acp.ndim(B)
        axes = [axes[0] % A_ndim, axes[1] % B_ndim]
        if argnum == 0:
            B_axes = np.arange(B_ndim)
            return acp.tensordot(g, B, [g_axes[A_ndim-1:], np.delete(B_axes, axes[1])])
        else:
            A_axes = np.arange(A_ndim)
            return acp.tensordot(A, g, [np.delete(A_axes, axes[0]), g_axes[:A_ndim-1]])
    else:
        B_ndim = acp.ndim(B)
        A_axes = np.arange(A_ndim)
        B_axes = np.arange(B_ndim)
        summed_axes = [np.asarray(axes[0]) % A_ndim,
                       np.asarray(axes[1]) % B_ndim]
        other_axes  = [np.delete(A_axes, summed_axes[0]),
                       np.delete(B_axes, summed_axes[1])]
        if argnum == 0:
            out = acp.tensordot(g, B, [g_axes[len(other_axes[0]):], other_axes[1]])
            perm = np.argsort(np.concatenate(
                (other_axes[0], summed_axes[0][np.argsort(summed_axes[1])])))
            return acp.transpose(out, perm)
        else:
            out = acp.tensordot(A, g, [other_axes[0], g_axes[:len(other_axes[0])]])
            perm = np.argsort(np.concatenate(
                (summed_axes[1][np.argsort(summed_axes[0])], other_axes[1])))
            return acp.transpose(out, perm)
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
