from __future__ import absolute_import
import operator as op
import autograd.numpy as np
from autograd.core import primitive, getval
import scipy.stats as sps

P = primitive

# ----- Operator gradients -----
I = lambda x : x # Identity operator
neg = P(op.neg, lambda ans, x    : [op.neg])
add = P(op.add, lambda ans, x, y : unbroadcast(ans, x, y, [I, I]))
mul = P(op.mul, lambda ans, x, y : unbroadcast(ans, x, y, [lambda g : y * g, lambda g : x * g]))
sub = P(op.sub, lambda ans, x, y : unbroadcast(ans, x, y, [I, op.neg]))
div = P(op.div, lambda ans, x, y : unbroadcast(ans, x, y, [lambda g : g / y, lambda g : - g * x / y**2]))
pow = P(op.pow, lambda ans, x, y : unbroadcast(ans, x, y, [lambda g : g * y * x ** (y - 1),
                                                           lambda g : g * np.log(x) * x ** y]))
isarray = lambda x : isinstance(getval(x), np.ndarray)
isfloat = lambda x : isinstance(getval(x), float)

def unbroadcast(ans, x, y, funs):
    return [unbroadcast_fun(ans, x, funs[0]),
            unbroadcast_fun(ans, y, funs[1])]

def unbroadcast_fun(ans, x, fun):
    if isfloat(x) and isarray(ans):
        return lambda g : np.sum(fun(g))
    elif isarray(x):
        shape = x.shape
        def new_fun(g):
            result = fun(g)
            while result.ndim > len(shape):
                result = np.sum(result, axis=0)
            for axis, size in enumerate(shape):
                if size is 1:
                    result = np.sum(result, axis, keepdims=True)
            return result
        return new_fun
    else:
        return fun

# ----- Scipy gradients -----

# TODO: wrap scipy too
sps.norm.cdf = P(sps.norm.cdf, lambda ans, x, loc=0.0, scale=1.0 : [lambda g : g * (1./(np.sqrt(2.0*np.pi)*scale)) *np.exp(-((x-loc)**2.0)/(2.0*(scale**2.)))])
