from __future__ import absolute_import

import autograd.numpy as np
import scipy.stats
from autograd.extend import primitive
from autograd.numpy.util import def_ufunc_jps
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

def_ufunc_jps(cdf,
              (lambda ans, x, a, b: np.power(x, a-1) * np.power(1-x, b-1) / beta(a, b), 'mul'),
              None,
              None)
def_ufunc_jps(logpdf,
              (lambda ans, x, a, b: grad_beta_logpdf_arg0(x, a, b), 'mul'),
              (lambda ans, x, a, b: grad_beta_logpdf_arg1(x, a, b), 'mul'),
              (lambda ans, x, a, b: grad_beta_logpdf_arg2(x, a, b), 'mul'))
def_ufunc_jps(pdf,
              (lambda ans, x, a, b: ans * grad_beta_logpdf_arg0(x, a, b), 'mul'),
              (lambda ans, x, a, b: ans * grad_beta_logpdf_arg1(x, a, b), 'mul'),
              (lambda ans, x, a, b: ans * grad_beta_logpdf_arg2(x, a, b), 'mul'))
