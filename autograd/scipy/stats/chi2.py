from __future__ import absolute_import, division

import autograd.numpy as np
import scipy.stats
from autograd.extend import primitive, defvjp
from autograd.numpy.numpy_vjps import unbroadcast_f
from autograd.scipy.special import gamma

cdf = primitive(scipy.stats.chi2.cdf)
logpdf = primitive(scipy.stats.chi2.logpdf)
pdf = primitive(scipy.stats.chi2.pdf)

def grad_chi2_logpdf(x, df):
    return np.where(df % 1 == 0, (df - x - 2) / (2 * x), 0)

defvjp(cdf, lambda ans, x, df: unbroadcast_f(x, lambda g: g * np.power(2., -df/2) * np.exp(-x/2) * np.power(x, df/2 - 1) / gamma(df/2)), argnums=[0])
defvjp(logpdf, lambda ans, x, df: unbroadcast_f(x, lambda g: g * grad_chi2_logpdf(x, df)), argnums=[0])
defvjp(pdf, lambda ans, x, df: unbroadcast_f(x, lambda g: g * ans * grad_chi2_logpdf(x, df)), argnums=[0])
