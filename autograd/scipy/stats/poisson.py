from __future__ import absolute_import

import autograd.numpy as np
import scipy.stats

from autograd.core import primitive
from autograd.scipy.special import digamma, gamma

cdf = primitive(scipy.stats.poisson.cdf)
logpmf = primitive(scipy.stats.poisson.logpmf)
pmf = primitive(scipy.stats.poisson.pmf)

def grad_poisson_logpdf(k, mu):
    return np.where(k % 1 == 0, k / mu - 1, 0)

cdf.defvjp(lambda g, ans, vs, gvs, k, mu: g * -pmf(np.floor(k), mu), argnum=1)
logpmf.defvjp(lambda g, ans, vs, gvs, k, mu: g * grad_poisson_logpdf(k, mu), argnum=1)
pmf.defvjp(lambda g, ans, vs, gvs, k, mu: g * ans * grad_poisson_logpdf(k, mu), argnum=1)
