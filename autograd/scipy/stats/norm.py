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

pdf.defgrad(lambda g, ans, x, loc=0.0, scale=1.0: unbroadcast(ans, x,     -g * ans * (x - loc) / scale**2))
pdf.defgrad(lambda g, ans, x, loc=0.0, scale=1.0: unbroadcast(ans, loc,    g * ans * (x - loc) / scale**2), argnum=1)
pdf.defgrad(lambda g, ans, x, loc=0.0, scale=1.0: unbroadcast(ans, scale,  g * ans * (((x - loc)/scale)**2 - 1.0)/scale), argnum=2)

cdf.defgrad(lambda g,ans, x, loc=0.0, scale=1.0: unbroadcast(ans, x,      g * pdf(x, loc, scale)))
cdf.defgrad(lambda g, ans, x, loc=0.0, scale=1.0: unbroadcast(ans, loc,   -g * pdf(x, loc, scale)), argnum=1)
cdf.defgrad(lambda g, ans, x, loc=0.0, scale=1.0: unbroadcast(ans, scale, -g * pdf(x, loc, scale)*(x-loc)/scale), argnum=2)

logpdf.defgrad(lambda g, ans, x, loc=0.0, scale=1.0: unbroadcast(ans, x,     -g * (x - loc) / scale**2))
logpdf.defgrad(lambda g, ans, x, loc=0.0, scale=1.0: unbroadcast(ans, loc,    g * (x - loc) / scale**2), argnum=1)
logpdf.defgrad(lambda g, ans, x, loc=0.0, scale=1.0: unbroadcast(ans, scale,  g * (-1.0/scale + (x - loc)**2/scale**3)), argnum=2)

logcdf.defgrad(lambda g, ans, x, loc=0.0, scale=1.0: unbroadcast(ans, x,      g * anp.exp(logpdf(x, loc, scale) - logcdf(x, loc, scale))))
logcdf.defgrad(lambda g, ans, x, loc=0.0, scale=1.0: unbroadcast(ans, loc,   -g * anp.exp(logpdf(x, loc, scale) - logcdf(x, loc, scale))), argnum=1)
logcdf.defgrad(lambda g, ans, x, loc=0.0, scale=1.0: unbroadcast(ans, scale, -g * anp.exp(logpdf(x, loc, scale) - logcdf(x, loc, scale))*(x-loc)/scale), argnum=2)
