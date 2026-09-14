"""Gradients of the normal distribution."""

import scipy.stats

import autograd.numpy as anp
from autograd.extend import defvjp, primitive
from autograd.numpy.numpy_vjps import unbroadcast_f

pdf = primitive(scipy.stats.norm.pdf)
cdf = primitive(scipy.stats.norm.cdf)
sf = primitive(scipy.stats.norm.sf)
logpdf = primitive(scipy.stats.norm.logpdf)
logcdf = primitive(scipy.stats.norm.logcdf)
logsf = primitive(scipy.stats.norm.logsf)
ppf = primitive(scipy.stats.norm.ppf)
isf = primitive(scipy.stats.norm.isf)

defvjp(
    pdf,
    lambda ans, x, loc=0.0, scale=1.0: unbroadcast_f(x, lambda g: -g * ans * (x - loc) / scale**2),
    lambda ans, x, loc=0.0, scale=1.0: unbroadcast_f(loc, lambda g: g * ans * (x - loc) / scale**2),
    lambda ans, x, loc=0.0, scale=1.0: unbroadcast_f(
        scale, lambda g: g * ans * (((x - loc) / scale) ** 2 - 1.0) / scale
    ),
)

defvjp(
    cdf,
    lambda ans, x, loc=0.0, scale=1.0: unbroadcast_f(x, lambda g: g * pdf(x, loc, scale)),
    lambda ans, x, loc=0.0, scale=1.0: unbroadcast_f(loc, lambda g: -g * pdf(x, loc, scale)),
    lambda ans, x, loc=0.0, scale=1.0: unbroadcast_f(
        scale, lambda g: -g * pdf(x, loc, scale) * (x - loc) / scale
    ),
)

defvjp(
    logpdf,
    lambda ans, x, loc=0.0, scale=1.0: unbroadcast_f(x, lambda g: -g * (x - loc) / scale**2),
    lambda ans, x, loc=0.0, scale=1.0: unbroadcast_f(loc, lambda g: g * (x - loc) / scale**2),
    lambda ans, x, loc=0.0, scale=1.0: unbroadcast_f(
        scale, lambda g: g * (-1.0 / scale + (x - loc) ** 2 / scale**3)
    ),
)

defvjp(
    logcdf,
    lambda ans, x, loc=0.0, scale=1.0: unbroadcast_f(
        x, lambda g: g * anp.exp(logpdf(x, loc, scale) - logcdf(x, loc, scale))
    ),
    lambda ans, x, loc=0.0, scale=1.0: unbroadcast_f(
        loc, lambda g: -g * anp.exp(logpdf(x, loc, scale) - logcdf(x, loc, scale))
    ),
    lambda ans, x, loc=0.0, scale=1.0: unbroadcast_f(
        scale, lambda g: -g * anp.exp(logpdf(x, loc, scale) - logcdf(x, loc, scale)) * (x - loc) / scale
    ),
)

defvjp(
    logsf,
    lambda ans, x, loc=0.0, scale=1.0: unbroadcast_f(
        x, lambda g: -g * anp.exp(logpdf(x, loc, scale) - logsf(x, loc, scale))
    ),
    lambda ans, x, loc=0.0, scale=1.0: unbroadcast_f(
        loc, lambda g: g * anp.exp(logpdf(x, loc, scale) - logsf(x, loc, scale))
    ),
    lambda ans, x, loc=0.0, scale=1.0: unbroadcast_f(
        scale, lambda g: g * anp.exp(logpdf(x, loc, scale) - logsf(x, loc, scale)) * (x - loc) / scale
    ),
)

defvjp(
    sf,
    lambda ans, x, loc=0.0, scale=1.0: unbroadcast_f(x, lambda g: -g * pdf(x, loc, scale)),
    lambda ans, x, loc=0.0, scale=1.0: unbroadcast_f(loc, lambda g: g * pdf(x, loc, scale)),
    lambda ans, x, loc=0.0, scale=1.0: unbroadcast_f(
        scale, lambda g: g * pdf(x, loc, scale) * (x - loc) / scale
    ),
)

defvjp(
    ppf,
    # ppf(q) is the inverse of cdf, so d/dq ppf(q) = 1 / pdf(ppf(q)). The pdf at
    # the shifted and scaled quantile already carries the 1 / scale factor, so
    # no extra scale factor belongs here.
    lambda ans, x, loc=0.0, scale=1.0: unbroadcast_f(x, lambda g: g / pdf(ppf(x, loc, scale), loc, scale)),
    # ppf(q, loc, scale) = loc + scale * z, with z independent of loc.
    lambda ans, x, loc=0.0, scale=1.0: unbroadcast_f(loc, lambda g: g),
    lambda ans, x, loc=0.0, scale=1.0: unbroadcast_f(scale, lambda g: g * (ans - loc) / scale),
)

defvjp(
    isf,
    # isf(q) = ppf(1 - q), which flips the sign of the q derivative.
    lambda ans, x, loc=0.0, scale=1.0: unbroadcast_f(x, lambda g: -g / pdf(isf(x, loc, scale), loc, scale)),
    lambda ans, x, loc=0.0, scale=1.0: unbroadcast_f(loc, lambda g: g),
    lambda ans, x, loc=0.0, scale=1.0: unbroadcast_f(scale, lambda g: g * (ans - loc) / scale),
)
