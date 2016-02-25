from __future__ import absolute_import
from functools import partial
import numpy as onp
import numpy.linalg as npla
from .numpy_wrapper import wrap_namespace
from . import numpy_wrapper as anp
from ..core import primitive
from builtins import range

wrap_namespace(npla.__dict__, globals())

# Some formulas are from
# "An extended collection of matrix derivative results
#  for forward and reverse mode algorithmic differentiation"
# by Mike Giles
# https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf

# transpose by swapping last two dimensions
T = lambda x: anp.swapaxes(x, -1, -2)

# add two dimensions to the end of x
add2d = lambda x: anp.array(x)[...,None,None]

det.defgrad(lambda ans, x: lambda g: add2d(g) * add2d(ans) * T(inv(x)))
slogdet.defgrad(lambda ans, x: lambda g: add2d(g[1]) * T(inv(x)))

def make_grad_inv(ans, x):
    dot = anp.dot if ans.ndim == 2 else partial(anp.einsum, '...ij,...jk->...ik')
    return lambda g: -dot(dot(T(ans), g), T(ans))
inv.defgrad(make_grad_inv)

def make_grad_solve(argnum, ans, a, b):
    updim = lambda x: x if x.ndim == a.ndim else x[...,None]
    dot = anp.dot if a.ndim == 2 else partial(anp.einsum, '...ij,...jk->...ik')

    grad_arg0 = lambda g: -dot(updim(solve(T(a), g)), T(updim(ans)))
    grad_arg1 = lambda g: solve(T(a), g)

    return grad_arg0 if argnum == 0 else grad_arg1
solve.defgrads(make_grad_solve, [0, 1])

def make_grad_norm(ans, x, ord=None, axis=None):
    def check_implemented():
        matrix_norm = (x.ndim==2 and axis is None) or isinstance(axis, tuple)
        frobenius_norm = ord is None or ord == 'fro'
        diffable_pnorm = ord is None or ord > 1

        if matrix_norm and not frobenius_norm:
            raise NotImplementedError(
                'Gradient of matrix norm not implemented for ord={}'.format(ord))
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
    N = x.shape[-1]
    w, v = ans              # Eigenvalues, eigenvectors.
    dot = anp.dot if x.ndim == 2 else partial(anp.einsum, '...ij,...jk->...ik')
    def eigh_grad(g):
        wg, vg = g          # Gradient w.r.t. eigenvalues, eigenvectors.
        w_repeated = anp.repeat(w[..., anp.newaxis], N, axis=-1)
        off_diag = anp.ones((N, N)) - anp.eye(N)
        F = off_diag / (T(w_repeated) - w_repeated + anp.eye(N))
        return dot(v * wg[..., anp.newaxis, :] + dot(v, F * dot(T(v), vg)), T(v))
    return eigh_grad
eigh.defgrad(make_grad_eigh)

def broadcasting_dsymv(alpha, A, x, lower=None):
    N = A.shape[-1]
    idxs = onp.arange(N)
    A_lower = onp.tril(A)
    A_sym = A_lower + T(A_lower)
    A_sym[..., idxs, idxs] *= 0.5
    return onp.einsum('...ij,...j->...i', A_sym, x)

def make_grad_cholesky(L, A):
    from ..scipy.linalg import solve_triangular as _solve_triangular
    from .. import numpy as np

    if A.ndim > 2 or L.ndim > 2: raise ValueError, 'broadcasting version not implemented'

    def solve_triangular(L, x, trans='N'):
        return _solve_triangular(L, x, lower=True, trans=trans)

    def conjugate_solve(L, X):
        'X -> L^{-T} X L^{-1}'
        return solve_triangular(L, solve_triangular(L, X.T, 'T').T, 'T')

    phi = lambda X: np.tril(X) / (1. + np.eye(X.shape[0]))

    @primitive
    def cholesky_grad(g):
        S = conjugate_solve(L, phi(np.dot(L.T, g)))
        return (S + S.T) / 2.
    return cholesky_grad
cholesky.defgrad(make_grad_cholesky)
