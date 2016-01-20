from __future__ import absolute_import
from functools import partial, wraps
import numpy as onp
import numpy.linalg as npla
from .numpy_wrapper import wrap_namespace, dot
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

def broadcasting_dsymv(alpha, A, x, lower=None):
    N = A.shape[-1]
    idxs = onp.arange(N)
    A_lower = onp.tril(A)
    A_sym = A_lower + T(A_lower)
    A_sym[..., idxs, idxs] *= 0.5
    return onp.einsum('...ij,...j->...i', A_sym, x)

def make_grad_cholesky(L, A):
    # based on choleskies_cython.pyx in SheffieldML/GPy and (Smith 1995)
    # TODO for higher-order differentiation, replace dsymv, get rid of inplace
    # ops, make cholesky grad primitive and defgrad? also ArrayNode assignment
    from scipy.linalg.blas import dsymv
    N = L.shape[-1]
    T = lambda x: onp.swapaxes(x, -1, -2)
    if A.ndim > 2:
        dsymv = broadcasting_dsymv
        dot = partial(onp.einsum, '...i,...i->...')
        diag = partial(onp.diagonal, axis1=-1, axis2=-2)
    else:
        dot = onp.dot
        diag = onp.diag

    @primitive
    def cholesky_grad(g):
        dL = onp.tril(g)
        dL[...,-1,-1] /= 2 * L[...,-1,-1]
        for k in range(N-2, -1, -1):
            dL[...,k+1:,k] -= dsymv(1., dL[...,k+1:,k+1:], L[...,k+1:,k], lower=True)
            dL[...,k+1:,k] -= diag(dL[...,k+1:,k+1:]) * L[...,k+1:,k]
            dL[...,k+1:,k] /= L[...,k:k+1,k]
            dL[...,k,k] -= dot(dL[...,k+1:,k], L[...,k+1:,k])
            dL[...,k,k] /= 2 * L[...,k,k]
        return (dL + T(dL))/2.
    return primitive(cholesky_grad)
cholesky.defgrad(make_grad_cholesky)
