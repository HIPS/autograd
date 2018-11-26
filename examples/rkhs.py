"""
Inferring a function from a reproducing kernel Hilbert space (RKHS) by taking
gradients of eval with respect to the function-valued argument
"""
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.extend import primitive, defvjp, defjvp, VSpace, Box
from autograd.util import func
from autograd import grad

class RKHSFun(object):
    def __init__(self, kernel, alphas={}):
        self.alphas = alphas
        self.kernel = kernel
        self.vs = RKHSFunVSpace(self)

    @primitive
    def __call__(self, x):
        return sum([a * self.kernel(x, x_repr)
                    for x_repr, a in self.alphas.items()], 0.0)

    def __add__(self, f):  return self.vs.add(self, f)
    def __mul__(self, a):  return self.vs.scalar_mul(self, a)

# TODO: add vjp of __call__ wrt x (and show it in action)
defvjp(func(RKHSFun.__call__),
       lambda ans, f, x: lambda g: RKHSFun(f.kernel, {x : 1}) * g)

class RKHSFunBox(Box, RKHSFun):
    @property
    def kernel(self): return self._value.kernel
RKHSFunBox.register(RKHSFun)

class RKHSFunVSpace(VSpace):
    def __init__(self, value):
        self.kernel = value.kernel

    def zeros(self): return RKHSFun(self.kernel)
    def randn(self):
        # These arbitrary vectors are not analogous to randn in any meaningful way
        N = npr.randint(1,3)
        return RKHSFun(self.kernel, dict(zip(npr.randn(N), npr.randn(N))))

    def _add(self, f, g):
        assert f.kernel is g.kernel
        return RKHSFun(f.kernel, add_dicts(f.alphas, g.alphas))

    def _scalar_mul(self, f, a):
        return RKHSFun(f.kernel, {x : a * a_cur for x, a_cur in f.alphas.items()})

    def _inner_prod(self, f, g):
        assert f.kernel is g.kernel
        return sum([a1 * a2 * f.kernel(x1, x2)
                    for x1, a1 in f.alphas.items()
                    for x2, a2 in g.alphas.items()], 0.0)
RKHSFunVSpace.register(RKHSFun)

def add_dicts(d1, d2):
    d = {}
    for k, v in d1.items() + d2.items():
        d[k] = d[k] + v if k in d else v
    return d

if __name__=="__main__":
    def sq_exp_kernel(x1, x2): return np.exp(-(x1-x2)**2)

    xs = range(5)
    ys = [1, 2, 3, 2, 1]

    def logprob(f, xs, ys):
        return -sum((f(x) - y)**2 for x, y in zip(xs, ys))

    f = RKHSFun(sq_exp_kernel)
    for i in range(100):
        f = f + grad(logprob)(f, xs, ys) * 0.01

    for x, y in zip(xs, ys):
        print('{}\t{}\t{}'.format(x, y, f(x)))
