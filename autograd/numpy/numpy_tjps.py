from __future__ import absolute_import
import numpy as onp
from functools import partial
from ..util import func  # TODO(mattjj): should this import use autograd.util, not ..util?
from autograd.tracer import primitive, getval
from autograd.vspace import vspace
from autograd.core import SparseObject
from autograd.tjp import deftjp, vjps_are_tjps
from . import numpy_wrapper as anp
from .numpy_boxes import ArrayBox

# ----- Binary ufuncs -----

# The only difference here is we have to use a modified unbroadcast function,
# which handles leading dimensions (if they exist). Otherwise, the expressions
# used in the VJPs already broadcast along leading dimensions of g.

deftjp(anp.add,         lambda ans, vs, out_vs, x, y : lambda g: unbroadcast(vs, out_vs, g))
deftjp(anp.add,         lambda ans, vs, out_vs, x, y : lambda g: unbroadcast(vs, out_vs, g), argnum=1)
deftjp(anp.multiply,    lambda ans, vs, out_vs, x, y : lambda g: unbroadcast(vs, out_vs, y * g))
deftjp(anp.multiply,    lambda ans, vs, out_vs, x, y : lambda g: unbroadcast(vs, out_vs, x * g), argnum=1)
deftjp(anp.subtract,    lambda ans, vs, out_vs, x, y : lambda g: unbroadcast(vs, out_vs, g))
deftjp(anp.subtract,    lambda ans, vs, out_vs, x, y : lambda g: unbroadcast(vs, out_vs, -g), argnum=1)
deftjp(anp.divide,      lambda ans, vs, out_vs, x, y : lambda g: unbroadcast(vs, out_vs,   g / y))
deftjp(anp.divide,      lambda ans, vs, out_vs, x, y : lambda g: unbroadcast(vs, out_vs, - g * x / y**2), argnum=1)
deftjp(anp.maximum,     lambda ans, vs, out_vs, x, y : lambda g: unbroadcast(vs, out_vs, g * balanced_eq(x, ans, y)))
deftjp(anp.maximum,     lambda ans, vs, out_vs, x, y : lambda g: unbroadcast(vs, out_vs, g * balanced_eq(y, ans, x)), argnum=1)
deftjp(anp.minimum,     lambda ans, vs, out_vs, x, y : lambda g: unbroadcast(vs, out_vs, g * balanced_eq(x, ans, y)))
deftjp(anp.minimum,     lambda ans, vs, out_vs, x, y : lambda g: unbroadcast(vs, out_vs, g * balanced_eq(y, ans, x)), argnum=1)
deftjp(anp.fmax,        lambda ans, vs, out_vs, x, y : lambda g: unbroadcast(vs, out_vs, g * balanced_eq(x, ans, y)))
deftjp(anp.fmax,        lambda ans, vs, out_vs, x, y : lambda g: unbroadcast(vs, out_vs, g * balanced_eq(y, ans, x)), argnum=1)
deftjp(anp.fmin,        lambda ans, vs, out_vs, x, y : lambda g: unbroadcast(vs, out_vs, g * balanced_eq(x, ans, y)))
deftjp(anp.fmin,        lambda ans, vs, out_vs, x, y : lambda g: unbroadcast(vs, out_vs, g * balanced_eq(y, ans, x)), argnum=1)
deftjp(anp.logaddexp,   lambda ans, vs, out_vs, x, y : lambda g: unbroadcast(vs, out_vs, g * anp.exp(x-ans)))
deftjp(anp.logaddexp,   lambda ans, vs, out_vs, x, y : lambda g: unbroadcast(vs, out_vs, g * anp.exp(y-ans)), argnum=1)
deftjp(anp.logaddexp2,  lambda ans, vs, out_vs, x, y : lambda g: unbroadcast(vs, out_vs, g * 2**(x-ans)))
deftjp(anp.logaddexp2,  lambda ans, vs, out_vs, x, y : lambda g: unbroadcast(vs, out_vs, g * 2**(y-ans)), argnum=1)
deftjp(anp.true_divide, lambda ans, vs, out_vs, x, y : lambda g: unbroadcast(vs, out_vs, g / y))
deftjp(anp.true_divide, lambda ans, vs, out_vs, x, y : lambda g: unbroadcast(vs, out_vs, - g * x / y**2), argnum=1)
deftjp(anp.mod,         lambda ans, vs, out_vs, x, y : lambda g: unbroadcast(vs, out_vs, g))
deftjp(anp.remainder,   lambda ans, vs, out_vs, x, y : lambda g: unbroadcast(vs, out_vs, g))
deftjp(anp.mod,         lambda ans, vs, out_vs, x, y : lambda g: unbroadcast(vs, out_vs, -g * anp.floor(x/y)), argnum=1)
deftjp(anp.remainder,   lambda ans, vs, out_vs, x, y : lambda g: unbroadcast(vs, out_vs, -g * anp.floor(x/y)), argnum=1)
deftjp(anp.power,
    lambda ans, vs, out_vs, x, y : lambda g:
    unbroadcast(vs, out_vs, g * y * x ** anp.where(y, y - 1, 1.)))
deftjp(anp.power,
    lambda ans, vs, out_vs, x, y : lambda g:
    unbroadcast(vs, out_vs, g * anp.log(replace_zero(x, 1.)) * x ** y), argnum=1)

# ----- Simple grads -----

# Some VJP implementations already broadcast along leading dimensions of g, so
# they work as TJP definitions too. We use the vjps_are_tjps function for that.

vjps_are_tjps(anp.absolute)
vjps_are_tjps(anp.reciprocal)
vjps_are_tjps(anp.exp)
vjps_are_tjps(anp.exp2)
vjps_are_tjps(anp.expm1)
vjps_are_tjps(anp.log)
vjps_are_tjps(anp.log2)
vjps_are_tjps(anp.log10)
vjps_are_tjps(anp.log1p)
vjps_are_tjps(anp.sin)
vjps_are_tjps(anp.cos)
vjps_are_tjps(anp.tan)
vjps_are_tjps(anp.arcsin)
vjps_are_tjps(anp.arccos)
vjps_are_tjps(anp.arctan)
vjps_are_tjps(anp.sinh)
vjps_are_tjps(anp.cosh)
vjps_are_tjps(anp.tanh)
vjps_are_tjps(anp.arcsinh)
vjps_are_tjps(anp.arccosh)
vjps_are_tjps(anp.arctanh)
vjps_are_tjps(anp.rad2deg)
vjps_are_tjps(anp.degrees)
vjps_are_tjps(anp.deg2rad)
vjps_are_tjps(anp.radians)
vjps_are_tjps(anp.square)
vjps_are_tjps(anp.sqrt)
vjps_are_tjps(anp.sinc)

vjps_are_tjps(anp.conj)
vjps_are_tjps(anp.conjugate)

# ----- Trickier grads -----

def tjp_dot_arg0(ans, vs, out_vs, A, B):
    if anp.ndim(B) == 0 or anp.ndim(B) == 1 or anp.ndim(A) == 0:
        contract_dims = max(0, anp.ndim(B) - (anp.ndim(A) != 0))
        return lambda G: anp.tensordot(G, B, contract_dims)
    else:
        return lambda G: anp.tensordot(G, anp.swapaxes(B, -1, -2), anp.ndim(B) - 1)
deftjp(anp.dot, tjp_dot_arg0)

def tjp_dot_arg1(ans, vs, out_vs, A, B):
    needs_transpose = anp.ndim(B) > 1 and anp.ndim(A) != 0
    swap = (lambda x: anp.swapaxes(x, -1, -2)) if needs_transpose else (lambda x: x)
    if anp.ndim(A) == 0 or anp.ndim(A) == 1 or anp.ndim(B) == 0:
        contract_dims = max(0, anp.ndim(A) - (anp.ndim(B) != 0))
        return lambda G: swap(anp.tensordot(G, A, contract_dims))
    else:
        return lambda G: swap(anp.tensordot(
            G, A, [range(-anp.ndim(A) - anp.ndim(B) + 2, -anp.ndim(B) + 1),
                   range(anp.ndim(A) - 1)]))
deftjp(anp.dot, tjp_dot_arg1, argnum=1)

def tjp_transpose(ans, in_vs, out_vs, x, axes=None):
    axes = tuple(reversed(range(in_vs.ndim))) if axes is None else anp.argsort(axes)
    return lambda g: anp.transpose(g, tuple(range(anp.ndim(g) - len(axes))) + axes)
deftjp(anp.transpose, tjp_transpose)

# ----- Utility functions -----

def unbroadcast(vs, out_vs, result):
    result_vs = vspace(result)
    leading_dims = result_vs.ndim - out_vs.ndim
    broadcast_idx = leading_dims
    while anp.ndim(result) > leading_dims + vs.ndim:
        result = anp.sum(result, axis=broadcast_idx)
    for axis, size in enumerate(vs.shape):
        if size == 1:
            result = anp.sum(result, axis=leading_dims + axis, keepdims=True)
    if result_vs.iscomplex and not vs.iscomplex:
        result = anp.real(result)
    return result

# ----- Extra functions used internally  -----

# TODO untake
