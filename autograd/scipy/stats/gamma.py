from __future__ import absolute_import

import autograd.numpy as np
import scipy.stats
from autograd.extend import primitive
from autograd.numpy.util import def_ufunc_jps
from autograd.scipy.special import gamma, psi

cdf = primitive(scipy.stats.gamma.cdf)
logpdf = primitive(scipy.stats.gamma.logpdf)
pdf = primitive(scipy.stats.gamma.pdf)

def grad_gamma_logpdf_arg0(x, a):
    return (a - x - 1) / x

def grad_gamma_logpdf_arg1(x, a):
    return np.log(x) - psi(a)

def_ufunc_jps(cdf,
              (lambda ans, x, a: np.exp(-x) * np.power(x, a-1) / gamma(a), 'mul'),
              None,
              None)
def_ufunc_jps(logpdf,
              (lambda ans, x, a: grad_gamma_logpdf_arg0(x, a), 'mul'),
              (lambda ans, x, a: grad_gamma_logpdf_arg1(x, a), 'mul'))
def_ufunc_jps(pdf,
              (lambda ans, x, a: ans * grad_gamma_logpdf_arg0(x, a), 'mul'),
              (lambda ans, x, a: ans * grad_gamma_logpdf_arg1(x, a), 'mul'))
