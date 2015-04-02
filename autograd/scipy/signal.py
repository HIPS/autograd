from __future__ import absolute_import
import scipy.signal

from autograd.core import getval, primitive

convolve = primitive(scipy.signal.convolve)


def make_grad_convolve_0(ans, in1, in2, mode='full'):
    if mode == 'full':
        return lambda g: convolve(g, in2[::-1], mode='valid')
    elif mode == 'same':
        return lambda g: convolve(g[::-1], in2, mode='same')[::-1]
    elif mode == 'valid':
        return lambda g: convolve(g, in2[::-1], mode='full')

def make_grad_convolve_1(ans, in1, in2, mode='full'):
    if mode == 'full':
        return lambda g: convolve(g[::-1], in1, mode='valid')[::-1]
    elif mode == 'same':
        return lambda g: convolve(in1, g[::-1], mode='same')[::-1]
    elif mode == 'valid':
        return lambda g: convolve(g[::-1], in1, mode='full')[::-1]

convolve.defgrad(make_grad_convolve_0, argnum=0)
convolve.defgrad(make_grad_convolve_1, argnum=1)

