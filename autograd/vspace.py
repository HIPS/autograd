from __future__ import absolute_import
import numpy as np
import numpy.random as npr
from .tracer import primitive, isbox

identity_vjp = lambda *args: lambda g: g

@primitive
def vs_add(vs, x_prev, x_new): return vs._add(x_prev, x_new)
vs_add.defvjps(identity_vjp, argnums=[1,2])

@primitive
def vs_mut_add(vs, x_prev, x_new): return vs._mut_add(x_prev, x_new)
vs_mut_add.defvjps(identity_vjp, argnums=[1,2])

@primitive
def vs_covector(vs, x): return vs._covector(x)
vs_covector.defvjp(lambda ans, vs, gvs, vs_, x: lambda g:
                   gvs.covector(g), argnum=1)

@primitive
def vs_scalar_mul(vs, x, a):
    return vs._scalar_mul(x, a)
vs_scalar_mul.defvjp(lambda ans, vs, gvs, vs_, x, a: lambda g:
                     vs.covector(gvs.scalar_mul(gvs.covector(g), a)), argnum=1)
vs_scalar_mul.defvjp(lambda ans, vs, gvs, vs_, x, a: lambda g:
                     gvs.inner_prod(g, gvs.covector(x)), argnum=2)

@primitive
def vs_inner_prod(vs, x, y):
    return vs._inner_prod(x, y)
vs_inner_prod.defvjp(lambda ans, vs, gvs, vs_, x, y: lambda g:
                     vs.covector(vs.scalar_mul(y, gvs.covector(g))), argnum=1)
vs_inner_prod.defvjp(lambda ans, vs, gvs, vs_, x, y: lambda g:
                     vs.covector(vs.scalar_mul(x, gvs.covector(g))), argnum=2)

class VSpace(object):
    __slots__ = []
    iscomplex = False
    def __init__(self, value):
        pass

    def zeros(self):          assert False
    def ones(self):           assert False
    def standard_basis(self): assert False
    def randn(self):          assert False

    add        = vs_add
    mut_add    = vs_mut_add
    scalar_mul = vs_scalar_mul
    inner_prod = vs_inner_prod
    covector   = vs_covector

    def _add(self, x, y):
        return x + y

    def _mut_add(self, x, y):
        x += y
        return x

    def _covector(self, x):
        return x

    def _scalar_mul(self, x, a):
        return x * a

    def _inner_prod(self, x, y):
        assert False

    def __eq__(self, other):
        return type(self) == type(other) and self.__dict__ == other.__dict__

    def __repr__(self):
        return "{}_{}".format(type(self).__name__, self.__dict__)

    def examples(self):
        # Used for testing only
        N = self.size
        unit_vect = np.zeros(N)
        unit_vect[npr.randint(N)] = 1.0
        unit_vect = self.unflatten(unit_vect)
        rand_vect = npr.randn(N)
        return [self.zeros(), self.unflatten(npr.randn(N))]

def vspace_flatten(value, covector=False):
    return vspace(value).flatten(value, covector)

def vspace(value):
    try:
        return vspace_mappings[type(value)](value)
    except KeyError:
        if isbox(value):
            return value.vspace
        else:
            raise TypeError("Can't find vspace for type {}".format(type(value)))

vspace_mappings = {}
def register_vspace(vspace_maker, value_type):
    vspace_mappings[value_type] = vspace_maker

def assert_vspace_match(x, expected_vspace):
    assert expected_vspace == vspace(x), \
        "\nGrad returned unexpected vector space" \
        "\nVector space is {}" \
        "\nExpected        {}".format(vspace(x), expected_vspace)
