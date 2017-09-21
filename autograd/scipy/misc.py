from __future__ import absolute_import
import scipy.misc
from autograd.extend import primitive, defvjp
import autograd.numpy as anp
from autograd.numpy.numpy_vjps import repeat_to_match_shape

logsumexp = primitive(scipy.misc.logsumexp)

def make_grad_logsumexp(ans, x, axis=None, b=1.0, keepdims=False):
    shape, dtype = anp.shape(x), anp.result_type(x)
    def vjp(g):
        g_repeated,   _ = repeat_to_match_shape(g,   shape, dtype, axis, keepdims)
        ans_repeated, _ = repeat_to_match_shape(ans, shape, dtype, axis, keepdims)
        return g_repeated * b * anp.exp(x - ans_repeated)
    return vjp

defvjp(logsumexp, make_grad_logsumexp)
