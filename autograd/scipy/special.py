from __future__ import absolute_import
import scipy.special
import autograd.numpy as np

from autograd.core import primitive

polygamma    = primitive(scipy.special.polygamma)
psi          = primitive(scipy.special.psi)        # psi(x) is just polygamma(0, x)
digamma      = primitive(scipy.special.digamma)    # digamma is another name for psi.
gamma        = primitive(scipy.special.gamma)
gammaln      = primitive(scipy.special.gammaln)
gammasgn     = primitive(scipy.special.gammasgn)
rgamma       = primitive(scipy.special.rgamma)
multigammaln = primitive(scipy.special.multigammaln)

gammasgn.defgrad_is_zero()
polygamma.defgrad_is_zero(argnums=(0,))
polygamma.defgrad(lambda ans, n, x: lambda g: g * polygamma(n + 1, x), argnum=1)
psi.defgrad(      lambda ans, x: lambda g: g * polygamma(1, x))
digamma.defgrad(  lambda ans, x: lambda g: g * polygamma(1, x))
gamma.defgrad(    lambda ans, x: lambda g: g * ans * psi(x))
gammaln.defgrad(  lambda ans, x: lambda g: g * psi(x))
rgamma.defgrad(   lambda ans, x: lambda g: g * psi(x) / -gamma(x))
multigammaln.defgrad(lambda ans, a, d:
    lambda g: g * np.sum(digamma(np.expand_dims(a, -1) - np.arange(d)/2.), -1))
multigammaln.defgrad_is_zero(argnums=(1,))


### Bessel functions ###

j0 = primitive(scipy.special.j0)
y0 = primitive(scipy.special.y0)
j1 = primitive(scipy.special.j1)
y1 = primitive(scipy.special.y1)
jn = primitive(scipy.special.jn)
yn = primitive(scipy.special.yn)

j0.defgrad(lambda ans, x: lambda g: -g * j1(x))
y0.defgrad(lambda ans, x: lambda g: -g * y1(x))
j1.defgrad(lambda ans, x: lambda g: g * (j0(x) - jn(2, x)) / 2.0)
y1.defgrad(lambda ans, x: lambda g: g * (y0(x) - yn(2, x)) / 2.0)
jn.defgrad_is_zero(argnums=(0,))
yn.defgrad_is_zero(argnums=(0,))
jn.defgrad(lambda ans, n, x: lambda g: g * (jn(n - 1, x) - jn(n + 1, x)) / 2.0, argnum=1)
yn.defgrad(lambda ans, n, x: lambda g: g * (yn(n - 1, x) - yn(n + 1, x)) / 2.0, argnum=1)


### Error Function ###
inv_root_pi = 0.56418958354775627928
erf = primitive(scipy.special.erf)
erfc = primitive(scipy.special.erfc)

erf.defgrad(lambda ans, x: lambda g: 2.*g*inv_root_pi*np.exp(-x**2))
erfc.defgrad(lambda ans, x: lambda g: -2.*g*inv_root_pi*np.exp(-x**2))
