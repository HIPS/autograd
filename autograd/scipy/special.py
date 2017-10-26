from __future__ import absolute_import
import scipy.special
import autograd.numpy as np
from autograd.numpy.util import def_ufunc_jps
from autograd.extend import primitive, defvjp

### Gamma functions ###
polygamma    = primitive(scipy.special.polygamma)
psi          = primitive(scipy.special.psi)        # psi(x) is just polygamma(0, x)
digamma      = primitive(scipy.special.digamma)    # digamma is another name for psi.
gamma        = primitive(scipy.special.gamma)
gammaln      = primitive(scipy.special.gammaln)
gammasgn     = primitive(scipy.special.gammasgn)
rgamma       = primitive(scipy.special.rgamma)
multigammaln = primitive(scipy.special.multigammaln)

defvjp(gammasgn, None)
def_ufunc_jps(polygamma, None, (lambda ans, n, x: polygamma(n + 1, x), 'mul'))
def_ufunc_jps(psi,      (lambda ans, x: polygamma(1, x), 'mul'))
defvjp(digamma,  lambda ans, x: lambda g: g * polygamma(1, x))
defvjp(gamma,    lambda ans, x: lambda g: g * ans * psi(x))
defvjp(gammaln,  lambda ans, x: lambda g: g * psi(x))
defvjp(rgamma,   lambda ans, x: lambda g: g * psi(x) / -gamma(x))
defvjp(multigammaln,lambda ans, a, d: lambda g:
       g * np.sum(digamma(np.expand_dims(a, -1) - np.arange(d)/2.), -1),
       None)

### Bessel functions ###
j0 = primitive(scipy.special.j0)
y0 = primitive(scipy.special.y0)
j1 = primitive(scipy.special.j1)
y1 = primitive(scipy.special.y1)
jn = primitive(scipy.special.jn)
yn = primitive(scipy.special.yn)

defvjp(j0,lambda ans, x: lambda g: -g * j1(x))
defvjp(y0,lambda ans, x: lambda g: -g * y1(x))
defvjp(j1,lambda ans, x: lambda g: g * (j0(x) - jn(2, x)) / 2.0)
defvjp(y1,lambda ans, x: lambda g: g * (y0(x) - yn(2, x)) / 2.0)
defvjp(jn, None, lambda ans, n, x: lambda g: g * (jn(n - 1, x) - jn(n + 1, x)) / 2.0)
defvjp(yn, None, lambda ans, n, x: lambda g: g * (yn(n - 1, x) - yn(n + 1, x)) / 2.0)

### Error Function ###
inv_root_pi = 0.56418958354775627928
erf = primitive(scipy.special.erf)
erfc = primitive(scipy.special.erfc)

defvjp(erf, lambda ans, x: lambda g:  2.*g*inv_root_pi*np.exp(-x**2))
defvjp(erfc,lambda ans, x: lambda g: -2.*g*inv_root_pi*np.exp(-x**2))


### Inverse error function ###
root_pi = 1.7724538509055159
erfinv = primitive(scipy.special.erfinv)
erfcinv = primitive(scipy.special.erfcinv)

defvjp(erfinv,lambda ans, x: lambda g: g * root_pi / 2 * np.exp(erfinv(x)**2))
defvjp(erfcinv,lambda ans, x: lambda g: -g * root_pi / 2 * np.exp(erfcinv(x)**2))

### Logit and Expit ###
logit = primitive(scipy.special.logit)
expit = primitive(scipy.special.expit)

defvjp(logit,lambda ans, x: lambda g: g / ( x * (1 - x)))
defvjp(expit,lambda ans, x: lambda g: g * ans * (1 - ans))
