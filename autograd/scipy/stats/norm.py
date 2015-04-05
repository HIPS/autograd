"""Gradients of the normal distribution."""

from __future__ import absolute_import
import scipy.stats

from autograd.core import primitive

pdf = primitive(scipy.stats.norm.pdf)
cdf = primitive(scipy.stats.norm.cdf)
logpdf = primitive(scipy.stats.norm.logpdf)

pdf.defgrad(lambda ans, x, loc=0.0, scale=1.0:
            lambda g: -g * ans * (x - loc) / scale**2)
pdf.defgrad(lambda ans, x, loc=0.0, scale=1.0:
            lambda g:  g * ans * (x - loc) / scale**2, argnum=1)
pdf.defgrad(lambda ans, x, loc=0.0, scale=1.0:
            lambda g:  g * ans * (((x - loc)/scale)**2 - 1.0)/scale, argnum=2)

cdf.defgrad(lambda ans, x, loc=0.0, scale=1.0:
            lambda g:  g * pdf(x, loc, scale))
cdf.defgrad(lambda ans, x, loc=0.0, scale=1.0:
            lambda g: -g * pdf(x, loc, scale), argnum=1)
cdf.defgrad(lambda ans, x, loc=0.0, scale=1.0:
            lambda g: -g * pdf(x, loc, scale)*(x-loc)/scale, argnum=2)

logpdf.defgrad(lambda ans, x, loc=0.0, scale=1.0:
               lambda g: -g * (x - loc) / scale**2)
logpdf.defgrad(lambda ans, x, loc=0.0, scale=1.0:
               lambda g:  g * (x - loc) / scale**2, argnum=1)
logpdf.defgrad(lambda ans, x, loc=0.0, scale=1.0:
               lambda g: g * (-1.0/scale + (x - loc)**2/scale**3), argnum=2)
