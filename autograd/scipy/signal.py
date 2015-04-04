from __future__ import absolute_import
import scipy.signal

from autograd.core import primitive
from autograd.numpy import flipud

convolve = primitive(scipy.signal.convolve)
convolve2d = primitive(scipy.signal.convolve2d)

def get_same_slice(L_in0, L_in1):
    left_pad = L_in0 - (L_in1 + 1) / 2
    return slice(left_pad, left_pad + L_in1)

def make_grad_convolve_0(ans, in0, in1, mode='full'):
    if mode == 'full':
        return lambda g: convolve(g, flipud(in1), mode='valid')
    elif mode == 'same':
        return lambda g: flipud(convolve(flipud(g), in1, mode='same'))
    elif mode == 'valid':
        return lambda g: convolve(g, flipud(in1), mode='full')
    else:
        raise Exception("Unrecognized mode {0}".format(mode))

convolve.defgrad(make_grad_convolve_0, argnum=0)

def make_grad_convolve_1(ans, in0, in1, mode='full'):
    if mode == 'full':
        return lambda g: convolve(g, flipud(in0), mode='valid')
    elif mode == 'same':
        idxs = get_same_slice(in0.shape[0], in1.shape[0])
        return lambda g : convolve(g, flipud(in0), mode='full')[idxs]
    elif mode == 'valid':
        return lambda g: convolve(flipud(in0), g, mode='valid')
    else:
        raise Exception("Unrecognized mode {0}".format(mode))

convolve.defgrad(make_grad_convolve_1, argnum=1)

def flip2d(x):
    return x[::-1, ::-1]

def make_grad_convolve2d_0(ans, in0, in1, mode='full'):
    if mode == 'full':
        return lambda g: convolve2d(g, flip2d(in1), mode='valid')
    elif mode == 'same':
        return lambda g: flip2d(convolve2d(flip2d(g), in1, mode='same'))
    elif mode == 'valid':
        return lambda g: convolve2d(g, flip2d(in1), mode='full')
    else:
        raise Exception("Unrecognized mode {0}".format(mode))

convolve2d.defgrad(make_grad_convolve2d_0, argnum=0)

def make_grad_convolve2d_1(ans, in0, in1, mode='full'):
    if mode == 'full':
        return lambda g: convolve2d(g, flip2d(in0), mode='valid')
    elif mode == 'same':
        idxs_0 = get_same_slice(in0.shape[0], in1.shape[0])
        idxs_1 = get_same_slice(in0.shape[1], in1.shape[1])
        return lambda g : convolve2d(g, flip2d(in0), mode='full')[idxs_0, idxs_1]
    elif mode == 'valid':
        return lambda g: flip2d(convolve2d(in0, flip2d(g), mode='valid'))
    else:
        raise Exception("Unrecognized mode {0}".format(mode))

convolve2d.defgrad(make_grad_convolve2d_1, argnum=1)
