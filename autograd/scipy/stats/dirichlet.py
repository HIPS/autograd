from __future__ import absolute_import
import scipy.stats

import autograd.numpy as np
from autograd.scipy.special import digamma
from autograd.core import primitive

rvs    = primitive(scipy.stats.dirichlet.rvs)
pdf    = primitive(scipy.stats.dirichlet.pdf)
logpdf = primitive(scipy.stats.dirichlet.logpdf)

logpdf.defgrad(lambda g, ans, x, alpha: g * (alpha - 1) / x, argnum=0)
logpdf.defgrad(lambda g, ans, x, alpha: g * (digamma(np.sum(alpha)) - digamma(alpha) + np.log(x)), argnum=1)

# Same as log pdf, but multiplied by the pdf (ans).
pdf.defgrad(lambda g, ans, x, alpha: g * ans * (alpha - 1) / x, argnum=0)
pdf.defgrad(lambda g, ans, x, alpha: g * ans * (digamma(np.sum(alpha)) - digamma(alpha) + np.log(x)), argnum=1)
