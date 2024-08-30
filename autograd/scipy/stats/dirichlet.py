import scipy.stats

import autograd.numpy as np
from autograd.extend import defvjp, primitive
from autograd.scipy.special import digamma

rvs = primitive(scipy.stats.dirichlet.rvs)
pdf = primitive(scipy.stats.dirichlet.pdf)
logpdf = primitive(scipy.stats.dirichlet.logpdf)

defvjp(
    logpdf,
    lambda ans, x, alpha: lambda g: g * (alpha - 1) / x,
    lambda ans, x, alpha: lambda g: g * (digamma(np.sum(alpha)) - digamma(alpha) + np.log(x)),
)

# Same as log pdf, but multiplied by the pdf (ans).
defvjp(
    pdf,
    lambda ans, x, alpha: lambda g: g * ans * (alpha - 1) / x,
    lambda ans, x, alpha: lambda g: g * ans * (digamma(np.sum(alpha)) - digamma(alpha) + np.log(x)),
)
