"""Gradients of the normal distribution."""

from __future__ import absolute_import
import scipy.stats
import autograd.numpy as anp

from autograd.core import primitive
from autograd.numpy.numpy_grads import unbroadcast

pdf = primitive(scipy.stats.norm.pdf)
cdf = primitive(scipy.stats.norm.cdf)
logpdf = primitive(scipy.stats.norm.logpdf)
logcdf = primitive(scipy.stats.norm.logcdf)

pdf.defvjp(lambda g, ans, vs, gvs, x, loc=0.0, scale=1.0: unbroadcast(vs, gvs, -g * ans * (x - loc) / scale**2))
pdf.defvjp(lambda g, ans, vs, gvs, x, loc=0.0, scale=1.0: unbroadcast(vs, gvs,  g * ans * (x - loc) / scale**2), argnum=1)
pdf.defvjp(lambda g, ans, vs, gvs, x, loc=0.0, scale=1.0: unbroadcast(vs, gvs,  g * ans * (((x - loc)/scale)**2 - 1.0)/scale), argnum=2)

cdf.defvjp(lambda g, ans, vs, gvs, x, loc=0.0, scale=1.0: unbroadcast(vs, gvs,  g * pdf(x, loc, scale)))
cdf.defvjp(lambda g, ans, vs, gvs, x, loc=0.0, scale=1.0: unbroadcast(vs, gvs, -g * pdf(x, loc, scale)), argnum=1)
cdf.defvjp(lambda g, ans, vs, gvs, x, loc=0.0, scale=1.0: unbroadcast(vs, gvs, -g * pdf(x, loc, scale)*(x-loc)/scale), argnum=2)

logpdf.defvjp(lambda g, ans, vs, gvs, x, loc=0.0, scale=1.0: unbroadcast(vs, gvs, -g * (x - loc) / scale**2))
logpdf.defvjp(lambda g, ans, vs, gvs, x, loc=0.0, scale=1.0: unbroadcast(vs, gvs,  g * (x - loc) / scale**2), argnum=1)
logpdf.defvjp(lambda g, ans, vs, gvs, x, loc=0.0, scale=1.0: unbroadcast(vs, gvs,  g * (-1.0/scale + (x - loc)**2/scale**3)), argnum=2)

logcdf.defvjp(lambda g, ans, vs, gvs, x, loc=0.0, scale=1.0: unbroadcast(vs, gvs,  g * anp.exp(logpdf(x, loc, scale) - logcdf(x, loc, scale))))
logcdf.defvjp(lambda g, ans, vs, gvs, x, loc=0.0, scale=1.0: unbroadcast(vs, gvs, -g * anp.exp(logpdf(x, loc, scale) - logcdf(x, loc, scale))), argnum=1)
logcdf.defvjp(lambda g, ans, vs, gvs, x, loc=0.0, scale=1.0: unbroadcast(vs, gvs, -g * anp.exp(logpdf(x, loc, scale) - logcdf(x, loc, scale))*(x-loc)/scale), argnum=2)
