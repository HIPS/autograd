"""Gradients of the normal distribution."""
from __future__ import absolute_import
import scipy.stats
import autograd.numpy as anp
from autograd.core import primitive, defvjp, defvjps, defvjp_is_zero, defvjp_argnum
from autograd.numpy.numpy_grads import unbroadcast

pdf = primitive(scipy.stats.norm.pdf)
cdf = primitive(scipy.stats.norm.cdf)
logpdf = primitive(scipy.stats.norm.logpdf)
logcdf = primitive(scipy.stats.norm.logcdf)

defvjp(pdf,lambda ans, vs, gvs, x, loc=0.0, scale=1.0: lambda g:
       unbroadcast(vs, gvs, -g * ans * (x - loc) / scale**2))
defvjp(pdf,lambda ans, vs, gvs, x, loc=0.0, scale=1.0: lambda g:
       unbroadcast(vs, gvs,  g * ans * (x - loc) / scale**2), argnum=1)
defvjp(pdf,lambda ans, vs, gvs, x, loc=0.0, scale=1.0: lambda g:
       unbroadcast(vs, gvs,  g * ans * (((x - loc)/scale)**2 - 1.0)/scale), argnum=2)

defvjp(cdf,lambda ans, vs, gvs, x, loc=0.0, scale=1.0: lambda g:
       unbroadcast(vs, gvs,  g * pdf(x, loc, scale)))
defvjp(cdf,lambda ans, vs, gvs, x, loc=0.0, scale=1.0: lambda g:
       unbroadcast(vs, gvs, -g * pdf(x, loc, scale)), argnum=1)
defvjp(cdf,lambda ans, vs, gvs, x, loc=0.0, scale=1.0: lambda g:
       unbroadcast(vs, gvs, -g * pdf(x, loc, scale)*(x-loc)/scale), argnum=2)

defvjp(logpdf,lambda ans, vs, gvs, x, loc=0.0, scale=1.0: lambda g:
       unbroadcast(vs, gvs, -g * (x - loc) / scale**2))
defvjp(logpdf,lambda ans, vs, gvs, x, loc=0.0, scale=1.0: lambda g:
       unbroadcast(vs, gvs,  g * (x - loc) / scale**2), argnum=1)
defvjp(logpdf,lambda ans, vs, gvs, x, loc=0.0, scale=1.0: lambda g:
       unbroadcast(vs, gvs,  g * (-1.0/scale + (x - loc)**2/scale**3)), argnum=2)

defvjp(logcdf,lambda ans, vs, gvs, x, loc=0.0, scale=1.0: lambda g:
       unbroadcast(vs, gvs,  g * anp.exp(logpdf(x, loc, scale) - logcdf(x, loc, scale))))
defvjp(logcdf,lambda ans, vs, gvs, x, loc=0.0, scale=1.0: lambda g:
       unbroadcast(vs, gvs, -g * anp.exp(logpdf(x, loc, scale) - logcdf(x, loc, scale))), argnum=1)
defvjp(logcdf,lambda ans, vs, gvs, x, loc=0.0, scale=1.0: lambda g:
       unbroadcast(vs, gvs, -g * anp.exp(logpdf(x, loc, scale) - logcdf(x, loc, scale))*(x-loc)/scale), argnum=2)
