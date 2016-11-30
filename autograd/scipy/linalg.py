from __future__ import division
import scipy.linalg

import autograd.numpy as anp
from autograd.numpy.numpy_wrapper import wrap_namespace

wrap_namespace(scipy.linalg.__dict__, globals())  # populates module namespace

sqrtm.defvjp(lambda g, ans, vs, gvs, A, **kwargs: solve_lyapunov(ans, g))

def _flip(a, trans):
    if anp.iscomplexobj(a):
        return 'H' if trans in ('N', 0) else 'N'
    else:
        return 'T' if trans in ('N', 0) else 'N'

def grad_solve_triangular(g, ans, vs, gvs, a, b, trans=0, lower=False, **kwargs):
    tri = anp.tril if (lower ^ (_flip(a, trans) == 'N')) else anp.triu
    transpose = lambda x: x if _flip(a, trans) != 'N' else x.T
    al2d = lambda x: x if x.ndim > 1 else x[...,None]
    v = al2d(solve_triangular(a, g, trans=_flip(a, trans), lower=lower))
    return -transpose(tri(anp.dot(v, al2d(ans).T)))

solve_triangular.defvjp(grad_solve_triangular)
solve_triangular.defvjp(lambda g, ans, vs, gvs, a, b, trans=0, lower=False, **kwargs:
    solve_triangular(a, g, trans=_flip(a, trans), lower=lower), argnum=1)
