from __future__ import absolute_import
import scipy.stats

import autograd.numpy as np
from autograd.scipy.special import digamma
from autograd.extend import primitive, defvjp

rvs    = primitive(scipy.stats.dirichlet.rvs)
pdf    = primitive(scipy.stats.dirichlet.pdf)
logpdf = primitive(scipy.stats.dirichlet.logpdf)

defvjp(logpdf,lambda ans, x, alpha: lambda g:
              g * (alpha - 1) / x,
              lambda ans, x, alpha: lambda g:
              g * (digamma(np.sum(alpha)) - digamma(alpha) + np.log(x)))

# Same as log pdf, but multiplied by the pdf (ans).
defvjp(pdf,lambda ans, x, alpha: lambda g:
           g * ans * (alpha - 1) / x,
           lambda ans, x, alpha: lambda g:
           g * ans * (digamma(np.sum(alpha)) - digamma(alpha) + np.log(x)))
