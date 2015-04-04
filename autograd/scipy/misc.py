from __future__ import absolute_import
import scipy.misc

from autograd.core import primitive
import autograd.numpy as anp
from autograd.numpy.numpy_grads import repeat_to_match_shape


logsumexp = primitive(scipy.misc.logsumexp)

def make_grad_logsumexp(ans, x, axis=None, b=1.0, keepdims=False):
    repeater, _ = repeat_to_match_shape(x, axis, keepdims)
    return lambda g: repeater(g) * b * anp.exp(x - repeater(ans))

logsumexp.defgrad(make_grad_logsumexp)