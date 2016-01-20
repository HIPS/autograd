"""Gradients of the univariate t distribution."""
from __future__ import absolute_import
import scipy.stats
import autograd.numpy as np

from autograd.core import primitive
from autograd.numpy.numpy_grads import unbroadcast
from autograd.scipy.special import psi

pdf = primitive(scipy.stats.t.pdf)
cdf = primitive(scipy.stats.t.cdf)
logpdf = primitive(scipy.stats.t.logpdf)
logcdf = primitive(scipy.stats.t.logcdf)

def grad_tlogpdf_diff(diff, df):
    return -diff * (1.0 + df) / (diff**2 + df)
def grad_tlogpdf_x(x, df, loc, scale):
    return grad_tlogpdf_diff((x - loc) / scale, df) / scale
def grad_tlogpdf_loc(x, df, loc, scale):
    return -grad_tlogpdf_diff((x - loc) / scale, df) / scale
def grad_tlogpdf_scale(x, df, loc, scale):
    diff = x - loc
    return -(df * (scale**2 - diff**2))/(scale * (df * scale**2 + diff**2))
def grad_tlogpdf_df(x, df, loc, scale):
    y = (x - loc)/scale
    return 0.5 * ((y**2 * (df+1))/(df * (y**2 + df)) - np.log(y**2 / df + 1) - 1.0/df -psi(df/2.0) + psi((df + 1)/2.0))

pdf.defgrad(lambda ans, x, df, loc=0.0, scale=1.0: unbroadcast(ans, x,     lambda g: g * ans * grad_tlogpdf_x(    x, df, loc, scale)), argnum=0)
pdf.defgrad(lambda ans, x, df, loc=0.0, scale=1.0: unbroadcast(ans, df,    lambda g: g * ans * grad_tlogpdf_df(   x, df, loc, scale)), argnum=1)
pdf.defgrad(lambda ans, x, df, loc=0.0, scale=1.0: unbroadcast(ans, loc,   lambda g: g * ans * grad_tlogpdf_loc(  x, df, loc, scale)), argnum=2)
pdf.defgrad(lambda ans, x, df, loc=0.0, scale=1.0: unbroadcast(ans, scale, lambda g: g * ans * grad_tlogpdf_scale(x, df, loc, scale)), argnum=3)

cdf.defgrad(lambda ans, x, df, loc=0.0, scale=1.0: unbroadcast(ans, x,     lambda g:  g * pdf(x, df, loc, scale)), argnum=0)
cdf.defgrad(lambda ans, x, df, loc=0.0, scale=1.0: unbroadcast(ans, loc,   lambda g: -g * pdf(x, df, loc, scale)), argnum=2)
# What is the gradient of the cdf wrt the degrees of freedom or scale?  No one knows.

logpdf.defgrad(lambda ans, x, df, loc=0.0, scale=1.0: unbroadcast(ans, x,     lambda g: g * grad_tlogpdf_x(    x, df, loc, scale)), argnum=0)
logpdf.defgrad(lambda ans, x, df, loc=0.0, scale=1.0: unbroadcast(ans, df,    lambda g: g * grad_tlogpdf_df(   x, df, loc, scale)), argnum=1)
logpdf.defgrad(lambda ans, x, df, loc=0.0, scale=1.0: unbroadcast(ans, loc,   lambda g: g * grad_tlogpdf_loc(  x, df, loc, scale)), argnum=2)
logpdf.defgrad(lambda ans, x, df, loc=0.0, scale=1.0: unbroadcast(ans, scale, lambda g: g * grad_tlogpdf_scale(x, df, loc, scale)), argnum=3)

logcdf.defgrad(lambda ans, x, df, loc=0.0, scale=1.0: unbroadcast(ans, x,     lambda g:  g * np.exp(logpdf(x, df, loc, scale) - logcdf(x, df, loc, scale))), argnum=0)
logcdf.defgrad(lambda ans, x, df, loc=0.0, scale=1.0: unbroadcast(ans, loc,   lambda g: -g * np.exp(logpdf(x, df, loc, scale) - logcdf(x, df, loc, scale))), argnum=2)
