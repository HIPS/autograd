from __future__ import absolute_import

import autograd.numpy as np
import scipy.stats
from autograd.extend import primitive, defvjp
from autograd.numpy.util import def_ufunc_jps

cdf = primitive(scipy.stats.poisson.cdf)
logpmf = primitive(scipy.stats.poisson.logpmf)
pmf = primitive(scipy.stats.poisson.pmf)

def grad_poisson_logpmf(k, mu):
    return np.where(k % 1 == 0, k / mu - 1, 0)

def_ufunc_jps(cdf,    None, (lambda ans, k, mu: -pmf(np.floor(k), mu),            'mul'))
def_ufunc_jps(logpmf, None, (lambda ans, k, mu: grad_poisson_logpmf(k, mu),       'mul'))
def_ufunc_jps(pmf,    None, (lambda ans, k, mu: ans * grad_poisson_logpmf(k, mu), 'mul'))
