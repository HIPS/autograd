from __future__ import absolute_import

import autograd.numpy as np
import scipy.stats
from autograd.extend import primitive, defvjp
from autograd.numpy.numpy_vjps import unbroadcast_f
from autograd.scipy.special import beta, psi

cdf = primitive(scipy.stats.beta.cdf)
logpdf = primitive(scipy.stats.beta.logpdf)
pdf = primitive(scipy.stats.beta.pdf)

def grad_beta_logpdf_arg0(x, a, b):
    return (1 + a * (x-1) + x * (b-2)) / (x * (x-1))

def grad_beta_logpdf_arg1(x, a, b):
    return np.log(x) - psi(a) + psi(a + b)

def grad_beta_logpdf_arg2(x, a, b):
    return np.log1p(-x) - psi(b) + psi(a + b)

defvjp(cdf, lambda ans, x, a, b: unbroadcast_f(x, lambda g: g * np.power(x, a-1) * np.power(1-x, b-1) / beta(a, b)), argnums=[0])
defvjp(logpdf,
       lambda ans, x, a, b: unbroadcast_f(x, lambda g: g * grad_beta_logpdf_arg0(x, a, b)),
       lambda ans, x, a, b: unbroadcast_f(a, lambda g: g * grad_beta_logpdf_arg1(x, a, b)),
       lambda ans, x, a, b: unbroadcast_f(b, lambda g: g * grad_beta_logpdf_arg2(x, a, b)))
defvjp(pdf,
       lambda ans, x, a, b: unbroadcast_f(x, lambda g: g * ans * grad_beta_logpdf_arg0(x, a, b)),
       lambda ans, x, a, b: unbroadcast_f(a, lambda g: g * ans * grad_beta_logpdf_arg1(x, a, b)),
       lambda ans, x, a, b: unbroadcast_f(b, lambda g: g * ans * grad_beta_logpdf_arg2(x, a, b)))
