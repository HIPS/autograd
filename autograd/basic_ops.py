from __future__ import absolute_import
import operator as op
from abc import ABCMeta
from numpy import log, ndarray, float16, float32, float64
from autograd.core import primitive, getval, Node
import scipy.stats as sps

# ----- Operator gradients -----

P = primitive
I = lambda x : x # Identity operator
neg = P(op.neg, lambda ans, x    : [op.neg])
add = P(op.add, lambda ans, x, y : unbroadcast[0](ans, x, y, [I, I]))
mul = P(op.mul, lambda ans, x, y : unbroadcast[0](ans, x, y, [lambda g : y * g, lambda g : x * g]))
sub = P(op.sub, lambda ans, x, y : unbroadcast[0](ans, x, y, [I, op.neg]))
div = P(op.div, lambda ans, x, y : unbroadcast[0](ans, x, y, [lambda g : g / y, lambda g : - g * x / y**2]))
pow = P(op.pow, lambda ans, x, y : unbroadcast[0](ans, x, y, [lambda g : g * y * x ** (y - 1),
                                                           lambda g : g * log(x) * x ** y]))
log = P(log,    lambda ans, x : [lambda g : g / x])

unbroadcast = []

# ----- Basic numeric node types -----

class NumericNode(Node):
    __array_priority__ = 100.0 # Ensure precedence of Node's __rmul__ over numpy's __mul__
    __metaclass__ = ABCMeta
    def __add__(self, other):   return add(self, other)
    def __radd__(self, other):  return add(self, other)
    def __sub__(self, other):   return sub(self, other)
    def __rsub__(self, other):  return sub(other, self)
    def __mul__(self, other):   return mul(self, other)
    def __rmul__(self, other):  return mul(other, self)
    def __neg__(self):          return neg(self)
    def __pow__(self, power):   return pow(self, power)
    def __rpow__(self, power):  return pow(power, self)
    def __div__(self, other):   return div(self, other)
    def __rdiv__(self, other):  return div(other, self)
    def __lt__(self, other):    return getval(self) < getval(other)
    def __gt__(self, other):    return getval(self) > getval(other) 

class FloatNode(NumericNode):
    pass

Node.add_subclass(FloatNode, [float, float16, float32, float64])

# ----- Scipy gradients -----

# # TODO: wrap scipy too
# sps.norm.cdf = P(sps.norm.cdf, lambda ans, x, loc=0.0, scale=1.0 : [lambda g : g * (1./(np.sqrt(2.0*np.pi)*scale)) *np.exp(-((x-loc)**2.0)/(2.0*(scale**2.)))])
