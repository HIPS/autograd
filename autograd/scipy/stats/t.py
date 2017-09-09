"""Gradients of the univariate t distribution."""
from __future__ import absolute_import
import scipy.stats
import autograd.numpy as np
from autograd.core import primitive, defvjp, defvjps, defvjp_is_zero, defvjp_argnum
from autograd.numpy.numpy_vjps import unbroadcast
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

defvjp(pdf,lambda ans, x, df, loc=0.0, scale=1.0: lambda g:
       unbroadcast(g * ans * grad_tlogpdf_x(    x, df, loc, scale)), argnum=0)
defvjp(pdf,lambda ans, x, df, loc=0.0, scale=1.0: lambda g:
       unbroadcast(g * ans * grad_tlogpdf_df(   x, df, loc, scale)), argnum=1)
defvjp(pdf,lambda ans, x, df, loc=0.0, scale=1.0: lambda g:
       unbroadcast(g * ans * grad_tlogpdf_loc(  x, df, loc, scale)), argnum=2)
defvjp(pdf,lambda ans, x, df, loc=0.0, scale=1.0: lambda g:
       unbroadcast(g * ans * grad_tlogpdf_scale(x, df, loc, scale)), argnum=3)

defvjp(cdf,lambda ans, x, df, loc=0.0, scale=1.0: lambda g:
       unbroadcast( g * pdf(x, df, loc, scale)), argnum=0)
defvjp(cdf,lambda ans, x, df, loc=0.0, scale=1.0: lambda g:
       unbroadcast(-g * pdf(x, df, loc, scale)), argnum=2)
# What is the gradient of the cdf wrt the degrees of freedom or scale?  No one knows.

defvjp(logpdf,lambda ans, x, df, loc=0.0, scale=1.0: lambda g:
       unbroadcast(g * grad_tlogpdf_x(    x, df, loc, scale)), argnum=0)
defvjp(logpdf,lambda ans, x, df, loc=0.0, scale=1.0: lambda g:
       unbroadcast(g * grad_tlogpdf_df(   x, df, loc, scale)), argnum=1)
defvjp(logpdf,lambda ans, x, df, loc=0.0, scale=1.0: lambda g:
       unbroadcast(g * grad_tlogpdf_loc(  x, df, loc, scale)), argnum=2)
defvjp(logpdf,lambda ans, x, df, loc=0.0, scale=1.0: lambda g:
       unbroadcast(g * grad_tlogpdf_scale(x, df, loc, scale)), argnum=3)

defvjp(logcdf,lambda ans, x, df, loc=0.0, scale=1.0: lambda g:
       unbroadcast( g * np.exp(logpdf(x, df, loc, scale) - logcdf(x, df, loc, scale))), argnum=0)
defvjp(logcdf,lambda ans, x, df, loc=0.0, scale=1.0: lambda g:
       unbroadcast(-g * np.exp(logpdf(x, df, loc, scale) - logcdf(x, df, loc, scale))), argnum=2)
