from __future__ import division
import scipy.linalg

import autograd.numpy as anp
from autograd.numpy.numpy_wrapper import wrap_namespace

wrap_namespace(scipy.linalg.__dict__, globals())  # populates module namespace

sqrtm.defgrad(lambda ans, A, **kwargs: lambda g: solve_lyapunov(ans, g))

def _flip(a, trans):
    if anp.iscomplexobj(a):
        return 'H' if trans in ('N', 0) else 'N'
    else:
        return 'T' if trans in ('N', 0) else 'N'

def make_grad_solve_triangular(ans, a, b, trans=0, lower=False, **kwargs):
    tri = anp.tril if (lower ^ (_flip(a, trans) == 'N')) else anp.triu
    transpose = lambda x: x if _flip(a, trans) != 'N' else x.T

    def solve_triangular_grad(g):
        al2d = lambda x: x if x.ndim > 1 else x[...,None]
        v = al2d(solve_triangular(a, g, trans=_flip(a, trans), lower=lower))
        return -transpose(tri(anp.dot(v, al2d(ans).T)))
    return solve_triangular_grad
solve_triangular.defgrad(make_grad_solve_triangular)
solve_triangular.defgrad(lambda ans, a, b, trans=0, lower=False, **kwargs: lambda g:
    solve_triangular(a, g, trans=_flip(a, trans), lower=lower), argnum=1)
