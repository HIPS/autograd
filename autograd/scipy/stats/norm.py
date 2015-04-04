from __future__ import absolute_import
import scipy.stats

from autograd.core import primitive

pdf = primitive(scipy.stats.norm.pdf)
cdf = primitive(scipy.stats.norm.cdf)

pdf.defgrad(lambda ans, x, loc=0.0, scale=1.0:
            lambda g: -g * (x - loc) / scale**2 * pdf(x, loc,scale))
pdf.defgrad(lambda ans, x, loc=0.0, scale=1.0:
            lambda g: g * (x - loc) / scale**2 * pdf(x, loc,scale), argnum=1)
cdf.defgrad(lambda ans, x, loc=0.0, scale=1.0: lambda g: g * pdf(x, loc, scale))
cdf.defgrad(lambda ans, x, loc=0.0, scale=1.0: lambda g: -g * pdf(x, loc, scale), argnum=2)
