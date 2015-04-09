from __future__ import absolute_import
import scipy.signal
from autograd.core import primitive
import autograd.numpy as np
import numpy as npo # original numpy
import itertools as it

def prod(x):
    return npo.prod(x, dtype=int)

@primitive
def convolve(A, B, axes=None, dot_axes=[(),()], mode='full'):
    """Generalization of scipy's convolve, which convolves over arbitrary axes,
    `axes`, and also allows a tensor dot over other axes, `dot_axes`."""
    # TODO: write (or borrow) a faster implementation. This is really just a placeholder.
    axes, shapes = parse_axes(A.shape, B.shape, axes, dot_axes, mode)
    if len(axes['A']['conv']) == 2:
        sp_convolve = scipy.signal.convolve2d
    else:
        sp_convolve = scipy.signal.convolve
    so = shapes['out']
    out = npo.zeros([prod(so['ignore_A']), prod(so['ignore_B'])] + so['conv'] )
    A = collect_axes(A, axes['A'], shapes['A'])
    B = collect_axes(B, axes['B'], shapes['B'])
    iterator = it.product(*[xrange(prod(s)) for s in
        [shapes['A']['ignore'], shapes['B']['ignore'], shapes['A']['dot']]])
    if any([a_size > b_size for a_size, b_size in zip(A.shape[2:], B.shape[2:])]):
        for i_A, i_B, i_dot in iterator:
            out[i_A, i_B] += sp_convolve(A[i_A, i_dot], B[i_B, i_dot], mode)
    else:
        for i_A, i_B, i_dot in iterator:
            out[i_A, i_B] += sp_convolve(B[i_B, i_dot], A[i_A, i_dot], mode)

    return out.reshape(so['ignore_A'] + so['ignore_B'] + so['conv'])

def collect_axes(X, axes, shapes):
    new_order = axes['ignore'] + axes['dot'] + axes['conv']
    new_shape = [prod(shapes['ignore']), prod(shapes['dot'])] + shapes['conv']
    return X.transpose(new_order).reshape(new_shape)

def parse_axes(A_shape, B_shape, conv_axes, dot_axes, mode):
    A_ndim, B_ndim = len(A_shape), len(B_shape)
    if conv_axes is None:
        conv_axes = [range(A_ndim), range(A_ndim)]
    axes = {'A' : {'conv' : list(conv_axes[0]),
                   'dot'  : list(dot_axes[0]),
                   'ignore' : [i for i in range(A_ndim)
                             if i not in conv_axes[0] and i not in dot_axes[0]]},
            'B' : {'conv' : list(conv_axes[1]),
                   'dot'  : list(dot_axes[1]),
                   'ignore' : [i for i in range(B_ndim)
                               if i not in conv_axes[1] and i not in dot_axes[1]]}}
    assert len(axes['A']['dot'])  == len(axes['B']['dot'])
    assert len(axes['A']['conv']) == len(axes['B']['conv'])
    i1 =      len(axes['A']['ignore'])
    i2 = i1 + len(axes['B']['ignore'])
    i3 = i2 + len(axes['A']['conv'])
    axes['out'] = {'ignore_A' : range(i1),
                   'ignore_B' : range(i1, i2),
                   'conv'     : range(i2, i3)}
    conv_shape = [compute_conv_size(A_shape[i], B_shape[j], mode)
                  for i, j in zip(axes['A']['conv'], axes['B']['conv'])]
    shapes = {'A'   : {s : [A_shape[i] for i in ax] for s, ax in axes['A'].iteritems()},
              'B'   : {s : [B_shape[i] for i in ax] for s, ax in axes['B'].iteritems()}}
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
    return new_idxs

def make_grad_convolve(argnum, ans, A, B, axes=None, dot_axes=[(),()], mode='full'):
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

    def grad_fun(g):
        result = convolve(g, Y[flipped_idxs(Y.ndim, axes[_Y_]['conv'])],
                          axes     = [axes['out']['conv'],   axes[_Y_]['conv']],
                          dot_axes = [axes['out'][ignore_Y], axes[_Y_]['ignore']],
                          mode     = new_mode)
        new_order = npo.argsort(axes[_X_]['ignore'] + axes[_X_]['dot'] + axes[_X_]['conv'])
        return np.transpose(result, new_order)
    return grad_fun
convolve.gradmaker = make_grad_convolve
