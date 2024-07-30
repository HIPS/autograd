from __future__ import absolute_import
from builtins import range, zip
from functools import partial
import autograd.numpy as np
import numpy as npo # original numpy
from autograd.extend import primitive, defvjp

from numpy.lib.stride_tricks import as_strided


@primitive
def convolve(A, B, axes=None, dot_axes=[(),()], mode='full'):
    assert mode in ['valid', 'full'], "Mode {0} not yet implemented".format(mode)
    if axes is None:
        axes = [list(range(A.ndim)), list(range(A.ndim))]
    wrong_order = any([B.shape[ax_B] < A.shape[ax_A] for ax_A, ax_B in zip(*axes)])
    if wrong_order:
        if mode=='valid' and not all([B.shape[ax_B] <= A.shape[ax_A] for ax_A, ax_B in zip(*axes)]):
                raise Exception("One array must be larger than the other along all convolved dimensions")
        elif mode != 'full' or B.size <= A.size: # Tie breaker
            i1 =      B.ndim - len(dot_axes[1]) - len(axes[1]) # B ignore
            i2 = i1 + A.ndim - len(dot_axes[0]) - len(axes[0]) # A ignore
            i3 = i2 + len(axes[0])
            ignore_B = list(range(i1))
            ignore_A = list(range(i1, i2))
            conv     = list(range(i2, i3))
            return convolve(B, A, axes=axes[::-1], dot_axes=dot_axes[::-1], mode=mode).transpose(ignore_A + ignore_B + conv)

    if mode == 'full':
        B = pad_to_full(B, A, axes[::-1])
    B_view_shape = list(B.shape)
    B_view_strides = list(B.strides)
    flipped_idxs = [slice(None)] * A.ndim
    for ax_A, ax_B in zip(*axes):
        B_view_shape.append(abs(B.shape[ax_B] - A.shape[ax_A]) + 1)
        B_view_strides.append(B.strides[ax_B])
        B_view_shape[ax_B] = A.shape[ax_A]
        flipped_idxs[ax_A] = slice(None, None, -1)
    B_view = as_strided(B, B_view_shape, B_view_strides)
    A_view = A[tuple(flipped_idxs)]
    all_axes = [list(axes[i]) + list(dot_axes[i]) for i in [0, 1]]
    return einsum_tensordot(A_view, B_view, all_axes)

def einsum_tensordot(A, B, axes, reverse=False):
    # Does tensor dot product using einsum, which shouldn't require a copy.
    A_axnums = list(range(A.ndim))
    B_axnums = list(range(A.ndim, A.ndim + B.ndim))
    sum_axnum = A.ndim + B.ndim
    for i_sum, (i_A, i_B) in enumerate(zip(*axes)):
        A_axnums[i_A] = sum_axnum + i_sum
        B_axnums[i_B] = sum_axnum + i_sum
    return npo.einsum(A, A_axnums, B, B_axnums)

def pad_to_full(A, B, axes):
    A_pad = [(0, 0)] * A.ndim
    for ax_A, ax_B in zip(*axes):
        A_pad[ax_A] = (B.shape[ax_B] - 1,) * 2
    return npo.pad(A, A_pad, mode='constant')

def parse_axes(A_shape, B_shape, conv_axes, dot_axes, mode):
    A_ndim, B_ndim = len(A_shape), len(B_shape)
    if conv_axes is None:
        conv_axes = (tuple(range(A_ndim)), tuple(range(A_ndim)),)
    axes = {'A' : {'conv' : tuple(conv_axes[0]),
                   'dot'  : tuple(dot_axes[0]),
                   'ignore' : tuple(i for i in range(A_ndim)
                             if i not in conv_axes[0] and i not in dot_axes[0])},
            'B' : {'conv' : tuple(conv_axes[1]),
                   'dot'  : tuple(dot_axes[1]),
                   'ignore' : tuple(i for i in range(B_ndim)
                               if i not in conv_axes[1] and i not in dot_axes[1])}}
    assert len(axes['A']['dot'])  == len(axes['B']['dot'])
    assert len(axes['A']['conv']) == len(axes['B']['conv'])
    i1 =      len(axes['A']['ignore'])
    i2 = i1 + len(axes['B']['ignore'])
    i3 = i2 + len(axes['A']['conv'])
    axes['out'] = {'ignore_A' : tuple(range(i1)),
                   'ignore_B' : tuple(range(i1, i2)),
                   'conv'     : tuple(range(i2, i3))}
    conv_shape = (compute_conv_size(A_shape[i], B_shape[j], mode)
                  for i, j in zip(axes['A']['conv'], axes['B']['conv']))
    shapes = {'A': {s: tuple(A_shape[i] for i in ax) for s, ax in axes['A'].items()},
              'B': {s: tuple(B_shape[i] for i in ax) for s, ax in axes['B'].items()}}
    shapes['out'] = {'ignore_A' : shapes['A']['ignore'],
                     'ignore_B' : shapes['B']['ignore'],
                     'conv'     : conv_shape}
    return axes, shapes

def compute_conv_size(A_size, B_size, mode):
    if mode == 'full':
        return A_size + B_size - 1
    elif mode == 'same':
        return A_size
    elif mode == 'valid':
        return abs(A_size - B_size) + 1
    else:
        raise Exception("Mode {0} not recognized".format(mode))

def flipped_idxs(ndim, axes):
    new_idxs = [slice(None)] * ndim
    for ax in axes:
        new_idxs[ax] = slice(None, None, -1)
    return tuple(new_idxs)

def grad_convolve(argnum, ans, A, B, axes=None, dot_axes=[(),()], mode='full'):
    assert mode in ['valid', 'full'], "Grad for mode {0} not yet implemented".format(mode)
    axes, shapes = parse_axes(A.shape, B.shape, axes, dot_axes, mode)
    if argnum == 0:
        X, Y = A, B
        _X_, _Y_ = 'A', 'B'
        ignore_Y = 'ignore_B'
    elif argnum == 1:
        X, Y = B, A
        _X_, _Y_ = 'B', 'A'
        ignore_Y = 'ignore_A'
    else:
        raise NotImplementedError("Can't take grad of convolve w.r.t. arg {0}".format(argnum))

    if mode == 'full':
        new_mode = 'valid'
    else:
        if any([x_size > y_size for x_size, y_size in zip(shapes[_X_]['conv'], shapes[_Y_]['conv'])]):
            new_mode = 'full'
        else:
            new_mode = 'valid'

    def vjp(g):
        result = convolve(g, Y[flipped_idxs(Y.ndim, axes[_Y_]['conv'])],
                          axes     = [axes['out']['conv'],   axes[_Y_]['conv']],
                          dot_axes = [axes['out'][ignore_Y], axes[_Y_]['ignore']],
                          mode     = new_mode)
        new_order = npo.argsort(axes[_X_]['ignore'] + axes[_X_]['dot'] + axes[_X_]['conv'])
        return np.transpose(result, new_order)
    return vjp

defvjp(convolve, partial(grad_convolve, 0), partial(grad_convolve, 1))
