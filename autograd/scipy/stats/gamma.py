import scipy.stats

import autograd.numpy as np
from autograd.extend import defvjp, primitive
from autograd.numpy.numpy_vjps import unbroadcast_f
from autograd.scipy.special import gamma, psi

cdf = primitive(scipy.stats.gamma.cdf)
logpdf = primitive(scipy.stats.gamma.logpdf)
pdf = primitive(scipy.stats.gamma.pdf)


def grad_gamma_logpdf_arg0(x, a):
    return (a - x - 1) / x


def grad_gamma_logpdf_arg1(x, a):
    return np.log(x) - psi(a)


defvjp(
    cdf,
    lambda ans, x, a: unbroadcast_f(x, lambda g: g * np.exp(-x) * np.power(x, a - 1) / gamma(a)),
    argnums=[0],
)
defvjp(
    logpdf,
    lambda ans, x, a: unbroadcast_f(x, lambda g: g * grad_gamma_logpdf_arg0(x, a)),
    lambda ans, x, a: unbroadcast_f(a, lambda g: g * grad_gamma_logpdf_arg1(x, a)),
)
defvjp(
    pdf,
    lambda ans, x, a: unbroadcast_f(x, lambda g: g * ans * grad_gamma_logpdf_arg0(x, a)),
    lambda ans, x, a: unbroadcast_f(a, lambda g: g * ans * grad_gamma_logpdf_arg1(x, a)),
)
