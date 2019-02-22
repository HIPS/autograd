"""Gradients of the normal distribution."""
from __future__ import absolute_import
import scipy.stats
import autograd.numpy as anp
from autograd.extend import primitive, defvjp
from autograd.numpy.numpy_vjps import unbroadcast_f

pdf = primitive(scipy.stats.norm.pdf)
cdf = primitive(scipy.stats.norm.cdf)
sf = primitive(scipy.stats.norm.sf)
logpdf = primitive(scipy.stats.norm.logpdf)
logcdf = primitive(scipy.stats.norm.logcdf)
logsf = primitive(scipy.stats.norm.logsf)

defvjp(pdf,
       lambda ans, x, loc=0.0, scale=1.0:
       unbroadcast_f(x, lambda g: -g * ans * (x - loc) / scale**2),
       lambda ans, x, loc=0.0, scale=1.0:
       unbroadcast_f(loc, lambda g: g * ans * (x - loc) / scale**2),
       lambda ans, x, loc=0.0, scale=1.0:
       unbroadcast_f(scale, lambda g: g * ans * (((x - loc)/scale)**2 - 1.0)/scale))

defvjp(cdf,
       lambda ans, x, loc=0.0, scale=1.0:
       unbroadcast_f(x, lambda g: g * pdf(x, loc, scale)) ,
       lambda ans, x, loc=0.0, scale=1.0:
       unbroadcast_f(loc, lambda g: -g * pdf(x, loc, scale)),
       lambda ans, x, loc=0.0, scale=1.0:
       unbroadcast_f(scale, lambda g: -g * pdf(x, loc, scale)*(x-loc)/scale))

defvjp(logpdf,
       lambda ans, x, loc=0.0, scale=1.0:
       unbroadcast_f(x, lambda g: -g * (x - loc) / scale**2),
       lambda ans, x, loc=0.0, scale=1.0:
       unbroadcast_f(loc, lambda g: g * (x - loc) / scale**2),
       lambda ans, x, loc=0.0, scale=1.0:
       unbroadcast_f(scale, lambda g: g * (-1.0/scale + (x - loc)**2/scale**3)))

defvjp(logcdf,
       lambda ans, x, loc=0.0, scale=1.0:
       unbroadcast_f(x, lambda g: g * anp.exp(logpdf(x, loc, scale) - logcdf(x, loc, scale))),
       lambda ans, x, loc=0.0, scale=1.0:
       unbroadcast_f(loc, lambda g:-g * anp.exp(logpdf(x, loc, scale) - logcdf(x, loc, scale))),
       lambda ans, x, loc=0.0, scale=1.0:
       unbroadcast_f(scale, lambda g:-g * anp.exp(logpdf(x, loc, scale) - logcdf(x, loc, scale))*(x-loc)/scale))

defvjp(logsf, 
       lambda ans, x, loc=0.0, scale=1.0: 
       unbroadcast_f(x, lambda g: -g * anp.exp(logpdf(x, loc, scale) - logsf(x, loc, scale))),
       lambda ans, x, loc=0.0, scale=1.0: 
       unbroadcast_f(loc, lambda g: g * anp.exp(logpdf(x, loc, scale) - logsf(x, loc, scale))),
       lambda ans, x, loc=0.0, scale=1.0: 
       unbroadcast_f(scale, lambda g: g * anp.exp(logpdf(x, loc, scale) - logsf(x, loc, scale)) * (x - loc) / scale))

defvjp(sf,
       lambda ans, x, loc=0.0, scale=1.0:
       unbroadcast_f(x, lambda g: -g * pdf(x, loc, scale)) ,
       lambda ans, x, loc=0.0, scale=1.0:
       unbroadcast_f(loc, lambda g: g * pdf(x, loc, scale)),
       lambda ans, x, loc=0.0, scale=1.0:
       unbroadcast_f(scale, lambda g: g * pdf(x, loc, scale)*(x-loc)/scale))
