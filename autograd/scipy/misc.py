from __future__ import absolute_import
import scipy.misc

from autograd.core import primitive
import autograd.numpy as anp
from autograd.numpy.numpy_grads import repeat_to_match_shape

logsumexp = primitive(scipy.misc.logsumexp)

def make_grad_logsumexp(g, ans, x, axis=None, b=1.0, keepdims=False):
    g_repeated,   _ = repeat_to_match_shape(g,   x, axis, keepdims)
    ans_repeated, _ = repeat_to_match_shape(ans, x, axis, keepdims)
    return g_repeated * b * anp.exp(x - ans_repeated)

logsumexp.defgrad(make_grad_logsumexp)
