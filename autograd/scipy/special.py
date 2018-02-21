from __future__ import absolute_import
import scipy.special
import autograd.numpy as np
from autograd.numpy.util import def_ufunc_jps, def_ufunc_jps_inv_pair
from autograd.extend import primitive

### Beta function ###
beta    = primitive(scipy.special.beta)
betainc = primitive(scipy.special.betainc)
betaln  = primitive(scipy.special.betaln)

def_ufunc_jps(beta,
              (lambda ans, a, b: ans * (psi(a) - psi(a + b)), 'mul'),
              (lambda ans, a, b: ans * (psi(b) - psi(a + b)), 'mul'))
def_ufunc_jps(betainc,
              None,
              None,
              (lambda ans, a, b, x: np.power(x, a - 1) * np.power(1 - x, b - 1) / beta(a, b), 'mul'))
def_ufunc_jps(betaln,
              (lambda ans, a, b: psi(a) - psi(a + b), 'mul'),
              (lambda ans, a, b: psi(b) - psi(a + b), 'mul'))

### Gamma functions ###
polygamma    = primitive(scipy.special.polygamma)
psi          = primitive(scipy.special.psi)        # psi(x) is just polygamma(0, x)
digamma      = primitive(scipy.special.digamma)    # digamma is another name for psi.
gamma        = primitive(scipy.special.gamma)
gammaln      = primitive(scipy.special.gammaln)
gammainc     = primitive(scipy.special.gammainc)
gammaincc    = primitive(scipy.special.gammaincc)
gammasgn     = primitive(scipy.special.gammasgn)
rgamma       = primitive(scipy.special.rgamma)
multigammaln = primitive(scipy.special.multigammaln)

def_ufunc_jps(gammasgn,  None)
def_ufunc_jps(polygamma, None, (lambda ans, n, x: polygamma(n + 1, x), 'mul'))
def_ufunc_jps(psi,       (lambda ans, x: polygamma(1, x),    'mul'))
def_ufunc_jps(digamma,   (lambda ans, x: polygamma(1, x),    'mul'))
def_ufunc_jps(gamma,     (lambda ans, x: ans * psi(x),       'mul'))
def_ufunc_jps(gammaln,   (lambda ans, x: psi(x),             'mul'))
def_ufunc_jps(rgamma,    (lambda ans, x: psi(x) / -gamma(x), 'mul'))
def_ufunc_jps(multigammaln, (lambda ans, a, d:
       np.sum(digamma(np.expand_dims(a, -1) - np.arange(d)/2.), -1), 'mul'),
       None)

def make_gammainc_vjp_arg1(sign):
    def gammainc_vjp_arg1(ans, a, x):
        return sign * np.exp(-x) * np.power(x, a - 1) / gamma(a)
    return gammainc_vjp_arg1
def_ufunc_jps(gammainc,  None, (make_gammainc_vjp_arg1(1),  'mul'))
def_ufunc_jps(gammaincc, None, (make_gammainc_vjp_arg1(-1), 'mul'))

### Bessel functions ###
j0 = primitive(scipy.special.j0)
y0 = primitive(scipy.special.y0)
j1 = primitive(scipy.special.j1)
y1 = primitive(scipy.special.y1)
jn = primitive(scipy.special.jn)
yn = primitive(scipy.special.yn)

def_ufunc_jps(j0, (lambda ans, x: -j1(x),                   'mul'))
def_ufunc_jps(y0, (lambda ans, x: -y1(x),                   'mul'))
def_ufunc_jps(j1, (lambda ans, x: (j0(x) - jn(2, x)) / 2.0, 'mul'))
def_ufunc_jps(y1, (lambda ans, x: (y0(x) - yn(2, x)) / 2.0, 'mul'))
def_ufunc_jps(jn, None, (lambda ans, n, x: (jn(n - 1, x) - jn(n + 1, x)) / 2.0, 'mul'))
def_ufunc_jps(yn, None, (lambda ans, n, x: (yn(n - 1, x) - yn(n + 1, x)) / 2.0, 'mul'))

### Error Function ###
inv_root_pi = 0.56418958354775627928
erf = primitive(scipy.special.erf)
erfc = primitive(scipy.special.erfc)

### Inverse error function ###
root_pi = 1.7724538509055159
erfinv = primitive(scipy.special.erfinv)
erfcinv = primitive(scipy.special.erfcinv)

def_ufunc_jps_inv_pair(erf,  erfinv,  lambda ans, x: 2.*inv_root_pi*np.exp(-x**2))
def_ufunc_jps_inv_pair(erfc, erfcinv, lambda ans, x: -2.*inv_root_pi*np.exp(-x**2))

### Logit and Expit ###
logit = primitive(scipy.special.logit)
expit = primitive(scipy.special.expit)

def_ufunc_jps_inv_pair(expit, logit, lambda ans, x:  ans * (1 - ans))

### Relative entropy ###
rel_entr = primitive(scipy.special.rel_entr)

def_ufunc_jps(rel_entr, (lambda ans, x, y: np.log(x / y) + 1, 'mul'), (lambda ans, x, y: - x / y, 'mul'))
