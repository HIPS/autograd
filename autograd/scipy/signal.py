from __future__ import absolute_import
import scipy.signal
from autograd.core import primitive
import autograd.numpy as np
import numpy as npo # original numpy
import itertools as it

prod = lambda x : npo.prod(x, dtype=int)

def either_order(convolve_fun):
    # scipy.signal.convolve requires A to be bigger than B for mode 'valid'. Not
    # quite sure why.
    def new_convolve(A, B, mode):
        if A.shape and A.shape[0] >= B.shape[0]:
            return convolve_fun(A, B, mode)
        else:
            return convolve_fun(B, A, mode)
    return new_convolve

@primitive
def convolve(A, B, axes=None, dot_axes=[(),()], mode='full'):
    """Generalization of scipy's convolve, which convolves over arbitrary axes,
    `axes`, and also allows a tensor dot over other axes, `dot_axes`."""
    # TODO: write (or borrow) a faster implementation. This is really just a placeholder.
    A_axes, B_axes, out_axes = parse_axes(A.ndim, B.ndim, axes, dot_axes)
    if len(A_axes['conv']) == 2:
        convolve_fun = either_order(scipy.signal.convolve2d)
    else:
        convolve_fun = either_order(scipy.signal.convolve)

    A_ignore_shape = [A.shape[i] for i in A_axes['ignore']]
    B_ignore_shape = [B.shape[i] for i in B_axes['ignore']]
    dot_shape      = [A.shape[i] for i in A_axes['dot']]
    conv_shape = [compute_conv_size(A.shape[i], B.shape[j], mode)
                  for i, j in zip(A_axes['conv'], B_axes['conv'])]
    out = npo.zeros([prod(A_ignore_shape), prod(B_ignore_shape)] + conv_shape)
    A = collect_axes(A, A_axes)
    B = collect_axes(B, B_axes)
    for i_A, i_B, i_dot in it.product(xrange(prod(A_ignore_shape)), 
                                      xrange(prod(B_ignore_shape)),
                                      xrange(prod(dot_shape))):
        out[i_A, i_B] += convolve_fun(A[i_A, i_dot], B[i_B, i_dot], mode)

    out = npo.reshape(out, A_ignore_shape + B_ignore_shape + conv_shape)
    return out

def compute_conv_size(A_size, B_size, mode):
    if mode == 'full':
        return A_size + B_size - 1
    elif mode == 'same':
        return A_size
    elif mode == 'valid':
        return abs(A_size - B_size) + 1
    else:
        raise Exception("Mode {0} not recognized".format(mode))

def parse_axes(A_ndim, B_ndim, axes, dot_axes):
    if axes is None:
        axes = [range(A_ndim), range(A_ndim)]
    A_axes = {'conv' : list(axes[0]),
              'dot'  : list(dot_axes[0]),
              'ignore' : [i for i in range(A_ndim)
                          if i not in axes[0] and i not in dot_axes[0]]}
    B_axes = {'conv' : list(axes[1]),
              'dot'  : list(dot_axes[1]),
              'ignore' : [i for i in range(B_ndim)
                          if i not in axes[1] and i not in dot_axes[1]]}
    assert len(A_axes['dot'])  == len(B_axes['dot'])
    assert len(A_axes['conv']) == len(B_axes['conv'])
    i1 = len(A_axes['ignore'])
    i2 = i1 + len(B_axes['ignore'])
    i3 = i2 + len(A_axes['conv'])
    out_axes = {'ignore_A' : range(i1),
                'ignore_B' : range(i1, i2),
                'conv'     : range(i2, i3)}
    return A_axes, B_axes, out_axes

def collect_axes(X, axes):
    result = npo.transpose(X, axes['ignore'] + axes['dot'] + axes['conv'])
    axes_shape = lambda name : [X.shape[s] for s in axes[name]]
    return npo.reshape(result, [prod(axes_shape('ignore')), prod(axes_shape('dot'))] + axes_shape('conv'))

def flipped_idxs(ndim, axes):
    new_idxs = [slice(None)] * ndim
    for ax in axes:
        new_idxs[ax] = slice(None, None, -1)
    return new_idxs

def make_grad_convolve(argnum, ans, A, B, axes=None, dot_axes=[(),()], mode='full'):
    assert mode in ['valid', 'full'], "Grad for mode {0} not yet implemented".format(mode)
    A_axes, B_axes, out_axes = parse_axes(A.ndim, B.ndim, axes, dot_axes)
    if argnum == 0:
        X, Y = A, B
        X_axes, Y_axes = A_axes, B_axes
        out_axes_ignore_Y = out_axes['ignore_B']
    elif argnum == 1:
        X, Y = B, A
        X_axes, Y_axes =  B_axes, A_axes
        out_axes_ignore_Y = out_axes['ignore_A']
    else:
        raise NotImplementedError("Can't take grad of convolve w.r.t. arg {0}".format(argnum))

    if mode == 'full' or X.shape[X_axes['conv'][0]] < Y.shape[Y_axes['conv'][0]]:
        new_mode = 'valid'
    else:
        new_mode = 'full'

    def grad_fun(g):
        result = convolve(g, Y[flipped_idxs(Y.ndim, Y_axes['conv'])],
                          axes     = [out_axes['conv'],  Y_axes['conv']],
                          dot_axes = [out_axes_ignore_Y, Y_axes['ignore']],
                          mode     = new_mode)
        new_order = npo.argsort(X_axes['ignore'] + X_axes['dot'] + X_axes['conv'])
        return np.transpose(result, new_order)
    return grad_fun
convolve.gradmaker = make_grad_convolve
