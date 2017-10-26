"""Gradients of the normal distribution."""
from __future__ import absolute_import
import scipy.stats
import autograd.numpy as anp
from autograd.extend import primitive, defvjp
from autograd.numpy.util import def_ufunc_jps

pdf = primitive(scipy.stats.norm.pdf)
cdf = primitive(scipy.stats.norm.cdf)
logpdf = primitive(scipy.stats.norm.logpdf)
logcdf = primitive(scipy.stats.norm.logcdf)

def_ufunc_jps(pdf,
    (lambda ans, x, loc=0.0, scale=1.0: -ans * (x - loc) / scale**2,              'mul'),
    (lambda ans, x, loc=0.0, scale=1.0: ans * (x - loc) / scale**2,               'mul'),
    (lambda ans, x, loc=0.0, scale=1.0: ans * (((x - loc)/scale)**2 - 1.0)/scale, 'mul'))

def_ufunc_jps(logpdf,
    (lambda ans, x, loc=0.0, scale=1.0: -(x - loc) / scale**2,              'mul'),
    (lambda ans, x, loc=0.0, scale=1.0: (x - loc) / scale**2,               'mul'),
    (lambda ans, x, loc=0.0, scale=1.0: -1.0/scale + (x - loc)**2/scale**3, 'mul'))

def_ufunc_jps(cdf,
    (lambda ans, x, loc=0.0, scale=1.0: pdf(x, loc, scale),                'mul'),
    (lambda ans, x, loc=0.0, scale=1.0: -pdf(x, loc, scale),               'mul'),
    (lambda ans, x, loc=0.0, scale=1.0: -pdf(x, loc, scale)*(x-loc)/scale, 'mul'))

def_ufunc_jps(logcdf,
    (lambda ans, x, loc=0.0, scale=1.0: anp.exp(logpdf(x, loc, scale) - logcdf(x, loc, scale)), 'mul'),
    (lambda ans, x, loc=0.0, scale=1.0:-anp.exp(logpdf(x, loc, scale) - logcdf(x, loc, scale)), 'mul'),
    (lambda ans, x, loc=0.0, scale=1.0:-anp.exp(logpdf(x, loc, scale) - logcdf(x, loc, scale))*(x-loc)/scale, 'mul'))
