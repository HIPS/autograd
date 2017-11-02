from __future__ import absolute_import, division

import autograd.numpy as np
import scipy.stats
from autograd.extend import primitive
from autograd.numpy.util import def_ufunc_jps
from autograd.scipy.special import gamma

cdf = primitive(scipy.stats.chi2.cdf)
logpdf = primitive(scipy.stats.chi2.logpdf)
pdf = primitive(scipy.stats.chi2.pdf)

def grad_chi2_logpdf(x, df):
    return np.where(df % 1 == 0, (df - x - 2) / (2 * x), 0)

def_ufunc_jps(cdf,    (lambda ans, x, df: (np.power(2., -df/2) * np.exp(-x/2) *
                       np.power(x, df/2 - 1) / gamma(df/2)), 'mul'), None)
def_ufunc_jps(logpdf, (lambda ans, x, df: (grad_chi2_logpdf(x, df)),       'mul'), None)
def_ufunc_jps(pdf,    (lambda ans, x, df: (ans * grad_chi2_logpdf(x, df)), 'mul'), None)
