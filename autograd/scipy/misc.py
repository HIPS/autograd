from __future__ import absolute_import
import scipy.misc

from autograd.core import primitive
import autograd.numpy as anp
from autograd.numpy.numpy_grads import repeat_to_match_shape

logsumexp = primitive(scipy.misc.logsumexp)

def make_grad_logsumexp(g, ans, vs, gvs, x, axis=None, b=1.0, keepdims=False):
    g_repeated,   _ = repeat_to_match_shape(g,   vs, axis, keepdims)
    ans_repeated, _ = repeat_to_match_shape(ans, vs, axis, keepdims)
    return g_repeated * b * anp.exp(x - ans_repeated)

logsumexp.defvjp(make_grad_logsumexp)

def make_fwd_grad_logsumexp(g, ans, gvs, vs, x, axis=None, b=1.0, keepdims=False):
    if not keepdims:
        if isinstance(axis, int):
            ans = anp.expand_dims(ans, axis)
        elif isinstance(axis, tuple):
            for ax in sorted(axis):
                ans = anp.expand_dims(ans, ax)
    return anp.sum(g * b * anp.exp(x - ans), axis=axis, keepdims=keepdims)

logsumexp.defjvp(make_fwd_grad_logsumexp)
