import scipy.stats

import autograd.numpy as np
from autograd.extend import defvjp, primitive
from autograd.numpy.numpy_vjps import unbroadcast_f
from autograd.scipy.special import digamma

rvs = primitive(scipy.stats.dirichlet.rvs)
pdf = primitive(scipy.stats.dirichlet.pdf)
logpdf = primitive(scipy.stats.dirichlet.logpdf)

defvjp(
    logpdf,
    lambda ans, x, alpha: lambda g: (np.expand_dims(g, -1) * (alpha - 1) / np.transpose(x)).T,
    lambda ans, x, alpha: unbroadcast_f(
        alpha,
        lambda g: (
            np.expand_dims(g, -1) * (digamma(np.sum(alpha)) - digamma(alpha) + np.log(np.transpose(x)))
        ),
    ),
)

# Same as log pdf, but multiplied by the pdf (ans).
defvjp(
    pdf,
    lambda ans, x, alpha: lambda g: (np.expand_dims(g * ans, -1) * (alpha - 1) / np.transpose(x)).T,
    lambda ans, x, alpha: unbroadcast_f(
        alpha,
        lambda g: (
            np.expand_dims(g * ans, -1) * (digamma(np.sum(alpha)) - digamma(alpha) + np.log(np.transpose(x)))
        ),
    ),
)
