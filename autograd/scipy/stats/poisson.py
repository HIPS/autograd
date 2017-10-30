from __future__ import absolute_import

import autograd.numpy as np
import scipy.stats
from autograd.extend import primitive, defvjp
from autograd.numpy.numpy_vjps import unbroadcast_f

cdf = primitive(scipy.stats.poisson.cdf)
logpmf = primitive(scipy.stats.poisson.logpmf)
pmf = primitive(scipy.stats.poisson.pmf)

def grad_poisson_logpmf(k, mu):
    return np.where(k % 1 == 0, k / mu - 1, 0)

defvjp(cdf, lambda ans, k, mu: unbroadcast_f(mu, lambda g: g * -pmf(np.floor(k), mu)), argnums=[1])
defvjp(logpmf, lambda ans, k, mu: unbroadcast_f(mu, lambda g: g * grad_poisson_logpmf(k, mu)), argnums=[1])
defvjp(pmf, lambda ans, k, mu: unbroadcast_f(mu, lambda g: g * ans * grad_poisson_logpmf(k, mu)), argnums=[1])
