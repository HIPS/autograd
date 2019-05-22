from __future__ import absolute_import
import scipy.special
import autograd.numpy as np
from autograd.extend import primitive, defvjp, defjvp
from autograd.numpy.numpy_vjps import unbroadcast_f, repeat_to_match_shape

### Beta function ###
beta    = primitive(scipy.special.beta)
betainc = primitive(scipy.special.betainc)
betaln  = primitive(scipy.special.betaln)

defvjp(beta,
       lambda ans, a, b: unbroadcast_f(a, lambda g: g * ans * (psi(a) - psi(a + b))),
       lambda ans, a, b: unbroadcast_f(b, lambda g: g * ans * (psi(b) - psi(a + b))))
defvjp(betainc,
       lambda ans, a, b, x: unbroadcast_f(x, lambda g: g * np.power(x, a - 1) * np.power(1 - x, b - 1) / beta(a, b)),
       argnums=[2])
defvjp(betaln,
       lambda ans, a, b: unbroadcast_f(a, lambda g: g * (psi(a) - psi(a + b))),
       lambda ans, a, b: unbroadcast_f(b, lambda g: g * (psi(b) - psi(a + b))))

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

defvjp(gammasgn, None)
defvjp(polygamma, None, lambda ans, n, x: lambda g: g * polygamma(n + 1, x))
defvjp(psi,      lambda ans, x: lambda g: g * polygamma(1, x))
defvjp(digamma,  lambda ans, x: lambda g: g * polygamma(1, x))
defvjp(gamma,    lambda ans, x: lambda g: g * ans * psi(x))
defvjp(gammaln,  lambda ans, x: lambda g: g * psi(x))
defvjp(rgamma,   lambda ans, x: lambda g: g * psi(x) / -gamma(x))
defvjp(multigammaln,lambda ans, a, d: lambda g:
       g * np.sum(digamma(np.expand_dims(a, -1) - np.arange(d)/2.), -1),
       None)

def make_gammainc_vjp_arg1(sign):
    def gammainc_vjp_arg1(ans, a, x):
        coeffs = sign * np.exp(-x) * np.power(x, a - 1) / gamma(a)
        return unbroadcast_f(x, lambda g: g * coeffs)
    return gammainc_vjp_arg1
defvjp(gammainc, make_gammainc_vjp_arg1(1), argnums=[1])
defvjp(gammaincc, make_gammainc_vjp_arg1(-1), argnums=[1])

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


### Faster versions of common Bessel functions ###
i0 = primitive(scipy.special.i0)
i1 = primitive(scipy.special.i1)
iv = primitive(scipy.special.iv)
ive = primitive(scipy.special.ive)

defvjp(i0, lambda ans, x: lambda g: g * i1(x))
defvjp(i1, lambda ans, x: lambda g: g * (i0(x) + iv(2, x)) / 2.0)
defvjp(iv,  None, lambda ans, n, x: lambda g: g * (iv(n - 1, x) + iv(n + 1, x)) / 2.0)
defvjp(ive, None, lambda ans, n, x: lambda g: g * (ans * (n / x - np.sign(x)) + ive(n + 1, x)))

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

### logsumexp ###
logsumexp = primitive(scipy.special.logsumexp)

def make_grad_logsumexp(ans, x, axis=None, b=1.0, keepdims=False):
    shape, dtype = np.shape(x), np.result_type(x)
    def vjp(g):
        g_repeated,   _ = repeat_to_match_shape(g,   shape, dtype, axis, keepdims)
        ans_repeated, _ = repeat_to_match_shape(ans, shape, dtype, axis, keepdims)
        return g_repeated * b * np.exp(x - ans_repeated)
    return vjp

defvjp(logsumexp, make_grad_logsumexp)

def fwd_grad_logsumexp(g, ans, x, axis=None, b=1.0, keepdims=False):
    if not keepdims:
        if isinstance(axis, int):
            ans = np.expand_dims(ans, axis)
        elif isinstance(axis, tuple):
            for ax in sorted(axis):
                ans = np.expand_dims(ans, ax)
    return np.sum(g * b * np.exp(x - ans), axis=axis, keepdims=keepdims)

defjvp(logsumexp, fwd_grad_logsumexp)
