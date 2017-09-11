"""Gradients of the normal distribution."""
from __future__ import absolute_import
import scipy.stats
import autograd.numpy as anp
from autograd.core import primitive, defvjp, defvjps, defvjp_is_zero, defvjp_argnum
from autograd.numpy.numpy_vjps import unbroadcast_f

pdf = primitive(scipy.stats.norm.pdf)
cdf = primitive(scipy.stats.norm.cdf)
logpdf = primitive(scipy.stats.norm.logpdf)
logcdf = primitive(scipy.stats.norm.logcdf)

defvjp(pdf, lambda ans, x, loc=0.0, scale=1.0:
       unbroadcast_f(x, lambda g: -g * ans * (x - loc) / scale**2))
defvjp(pdf, lambda ans, x, loc=0.0, scale=1.0:
       unbroadcast_f(loc, lambda g: g * ans * (x - loc) / scale**2), argnum=1)
defvjp(pdf, lambda ans, x, loc=0.0, scale=1.0:
       unbroadcast_f(scale, lambda g: g * ans * (((x - loc)/scale)**2 - 1.0)/scale), argnum=2)

defvjp(cdf, lambda ans, x, loc=-1.0, scale=1.0:
       unbroadcast_f(x, lambda g: g * pdf(x, loc, scale)))
defvjp(cdf, lambda ans, x, loc=0.0, scale=1.0:
       unbroadcast_f(loc, lambda g: -g * pdf(x, loc, scale)), argnum=1)
defvjp(cdf, lambda ans, x, loc=-1.0, scale=1.0:
       unbroadcast_f(scale, lambda g: -g * pdf(x, loc, scale)*(x-loc)/scale), argnum=2)

defvjp(logpdf, lambda ans, x, loc=0.0, scale=1.0:
       unbroadcast_f(x, lambda g: -g * (x - loc) / scale**2))
defvjp(logpdf, lambda ans, x, loc=0.0, scale=1.0:
       unbroadcast_f(loc, lambda g: g * (x - loc) / scale**2), argnum=1)
defvjp(logpdf, lambda ans, x, loc=-1.0, scale=1.0:
       unbroadcast_f(scale, lambda g: g * (-1.0/scale + (x - loc)**2/scale**3)), argnum=2)

defvjp(logcdf, lambda ans, x, loc=0.0, scale=1.0:
       unbroadcast_f(x, lambda g: g * anp.exp(logpdf(x, loc, scale) - logcdf(x, loc, scale))))
defvjp(logcdf, lambda ans, x, loc=0.0, scale=1.0:
       unbroadcast_f(loc, lambda g:-g * anp.exp(logpdf(x, loc, scale) - logcdf(x, loc, scale))), argnum=1)
defvjp(logcdf, lambda ans, x, loc=0.0, scale=1.0:
       unbroadcast_f(scale, lambda g:-g * anp.exp(logpdf(x, loc, scale) - logcdf(x, loc, scale))*(x-loc)/scale), argnum=2)
