from __future__ import absolute_import
import numpy.linalg as npla
from .numpy_wrapper import wrap_namespace, dot
from . import numpy_wrapper as anp

wrap_namespace(npla.__dict__, globals())

def atleast_2d_col(x):
    # Promotes a 1D array into a column rather than a row.
    return x if x.ndim > 1 else x[:,None]

# Some formulas are from
# "An extended collection of matrix derivative results
#  for forward and reverse mode algorithmic differentiation"
# by Mike Giles
# https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
inv.defgrad(    lambda ans, x    : lambda g : -dot(dot(ans.T, g), ans.T))
det.defgrad(    lambda ans, x    : lambda g : g * ans * inv(x).T)
slogdet.defgrad(lambda ans, x    : lambda g : g[1] * inv(x).T)
solve.defgrad(  lambda ans, a, b : lambda g : -dot(atleast_2d_col(solve(a.T, g)),
                                                 atleast_2d_col(ans).T))
solve.defgrad(lambda ans, a, b : lambda g : solve(a.T, g), argnum=1)

def make_grad_norm(ans, x, ord=None, axis=None):
    def check_implemented():
        matrix_norm = (x.ndim==2 and axis is None) or isinstance(axis, tuple)
        frobenius_norm = ord is None or ord == 'fro'
        diffable_pnorm = ord is None or ord > 1

        if matrix_norm and not frobenius_norm:
            raise NotImplementedError(
                'Gradient of norm not implemented for matrix norms other than fro')
        if not diffable_pnorm:
            raise NotImplementedError(
                'Gradient of norm not implemented for ord={}'.format(ord))

    expand = lambda a: a if axis is None else anp.expand_dims(a, axis=axis)

    def norm_grad(g):
        check_implemented()
        if ord is None or ord == 2 or ord is 'fro':
            return expand(g / ans) * x
        else:
            # see https://en.wikipedia.org/wiki/Norm_(mathematics)#p-norm
            return expand(g / ans**(ord-1)) * x * anp.abs(x)**(ord-2)
    return norm_grad
norm.defgrad(make_grad_norm)

def make_grad_eigh(ans, x, UPLO='L'):
    """Gradient for eigenvalues and vectors of a symmetric matrix."""
    N = x.shape[0]
    w, v = ans              # Eigenvalues, eigenvectors.
    def eigh_grad(g):
        wg, vg = g          # Gradient w.r.t. eigenvalues, eigenvectors.
        w_repeated = anp.repeat(w[:, anp.newaxis], N, 1)
        off_diag = anp.ones((N, N)) - anp.eye(N)
        F = off_diag / (w_repeated.T - w_repeated + anp.eye(N))
        dx = dot(v * wg + dot(v, F * dot(v.T, vg)), v.T)
        if UPLO == 'U':     # Reflect to account for symmetry.
            return anp.triu(dx) + anp.tril(dx, -1).T
        else:
            return anp.tril(dx) + anp.triu(dx, 1).T
    return eigh_grad
eigh.defgrad(make_grad_eigh)
