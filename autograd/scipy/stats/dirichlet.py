from __future__ import absolute_import
import scipy.stats

import autograd.numpy as np
from autograd.scipy.special import digamma
from autograd.core import primitive

rvs    = primitive(scipy.stats.dirichlet.rvs)
pdf    = primitive(scipy.stats.dirichlet.pdf)
logpdf = primitive(scipy.stats.dirichlet.logpdf)

logpdf.defvjp(lambda ans, vs, gvs, x, alpha: lambda g:
              g * (alpha - 1) / x, argnum=0)
logpdf.defvjp(lambda ans, vs, gvs, x, alpha: lambda g:
              g * (digamma(np.sum(alpha)) - digamma(alpha) + np.log(x)), argnum=1)

# Same as log pdf, but multiplied by the pdf (ans).
pdf.defvjp(lambda ans, vs, gvs, x, alpha: lambda g:
           g * ans * (alpha - 1) / x, argnum=0)
pdf.defvjp(lambda ans, vs, gvs, x, alpha: lambda g:
           g * ans * (digamma(np.sum(alpha)) - digamma(alpha) + np.log(x)), argnum=1)
