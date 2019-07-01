from __future__ import absolute_import
from functools import partial
import numpy.linalg as npla
from .numpy_wrapper import wrap_namespace
from . import numpy_wrapper as anp
from autograd.extend import defvjp

wrap_namespace(npla.__dict__, globals())

# Some formulas are from
# "An extended collection of matrix derivative results
#  for forward and reverse mode algorithmic differentiation"
# by Mike Giles
# https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf

# transpose by swapping last two dimensions
def T(x): return anp.swapaxes(x, -1, -2)

_dot = partial(anp.einsum, '...ij,...jk->...ik')

# add two dimensions to the end of x
def add2d(x): return anp.reshape(x, anp.shape(x) + (1, 1))

defvjp(det, lambda ans, x: lambda g: add2d(g) * add2d(ans) * T(inv(x)))
defvjp(slogdet, lambda ans, x: lambda g: add2d(g[1]) * T(inv(x)))

def grad_inv(ans, x):
    return lambda g: -_dot(_dot(T(ans), g), T(ans))
defvjp(inv, grad_inv)

def grad_pinv(ans, x):
    # https://mathoverflow.net/questions/25778/analytical-formula-for-numerical-derivative-of-the-matrix-pseudo-inverse
    return lambda g: T(
        -_dot(_dot(ans, T(g)), ans)
        + _dot(_dot(_dot(ans, T(ans)), g), anp.eye(x.shape[-2]) - _dot(x,ans))
        + _dot(_dot(_dot(anp.eye(ans.shape[-2]) - _dot(ans,x), g), T(ans)), ans)
        )
defvjp(pinv, grad_pinv)

def grad_solve(argnum, ans, a, b):
    updim = lambda x: x if x.ndim == a.ndim else x[...,None]
    if argnum == 0:
        return lambda g: -_dot(updim(solve(T(a), g)), T(updim(ans)))
    else:
        return lambda g: solve(T(a), g)
defvjp(solve, partial(grad_solve, 0), partial(grad_solve, 1))

def grad_norm(ans, x, ord=None, axis=None):
    def check_implemented():
        matrix_norm = (x.ndim == 2 and axis is None) or isinstance(axis, tuple)

        if matrix_norm:
            if not (ord is None or ord == 'fro' or ord == 'nuc'):
                raise NotImplementedError('Gradient of matrix norm not '
                                          'implemented for ord={}'.format(ord))
        elif not (ord is None or ord > 1):
            raise NotImplementedError('Gradient of norm not '
                                      'implemented for ord={}'.format(ord))

    if axis is None:
        expand = lambda a: a
    elif isinstance(axis, tuple):
        row_axis, col_axis = axis
        if row_axis > col_axis:
            row_axis = row_axis - 1
        expand = lambda a: anp.expand_dims(anp.expand_dims(a,
                                                   row_axis), col_axis)
    else:
        expand = lambda a: anp.expand_dims(a, axis=axis)

    if ord == 'nuc':
        if axis is None:
            roll = lambda a: a
            unroll = lambda a: a
        else:
            row_axis, col_axis = axis
            if row_axis > col_axis:
                row_axis = row_axis - 1
            # Roll matrix axes to the back
            roll = lambda a: anp.rollaxis(anp.rollaxis(a, col_axis, a.ndim),
                                          row_axis, a.ndim-1)
            # Roll matrix axes to their original position
            unroll = lambda a: anp.rollaxis(anp.rollaxis(a, a.ndim-2, row_axis),
                                            a.ndim-1, col_axis)

    # Used for returning zero gradient of zero norms
    replace_zero     = lambda x, val: anp.where(x, x, val)
    # For manually set the second derivative of norm to zero, to match np.abs()
    replace_zero_ans = lambda x, val: anp.where(expand(ans), x, val)

    check_implemented()
    def vjp(g):
        if ord is None or ord == 2 or ord is 'fro':
            # The gradient is 1 / ans * x
            # FIXME: when x is complex vector, the gradient seems to be:
            #        1 / ans * conj(x)
            return expand(g / replace_zero(ans, 1.)) * replace_zero_ans(x, 0.)
        elif ord == 'nuc':
            x_rolled = roll(x)
            u, s, vt = svd(x_rolled, full_matrices=False)
            uvt_rolled = _dot(u, vt)
            # Roll the matrix axes back to their correct positions
            uvt = unroll(uvt_rolled)
            g = expand(g)
            return g * uvt
        else:
            # See https://en.wikipedia.org/wiki/Norm_(mathematics)#p-norm
            # The gradient is 1 / ans**(ord-1) * abs(x)**(ord-1) * sign(x)
            # Use `abs(x)**(ord-1) * sign(x)` instead of `abs(x)**(ord-2) * x`
            # avoids NaN when x contains zero.
            # FIXME: when x is complex vector, the gradient seems to be:
            #        1 / ans**(ord-1) * abs(x)**(ord-1) * conj(x) / abs(x)
            return expand(g / replace_zero(ans**(ord-1), 1.0)) \
                   * anp.abs(x)**(ord-1) * anp.sign(x)
    return vjp
defvjp(norm, grad_norm)

def grad_eigh(ans, x, UPLO='L'):
    """Gradient for eigenvalues and vectors of a symmetric matrix."""
    N = x.shape[-1]
    w, v = ans              # Eigenvalues, eigenvectors.
    def vjp(g):
        wg, vg = g          # Gradient w.r.t. eigenvalues, eigenvectors.
        w_repeated = anp.repeat(w[..., anp.newaxis], N, axis=-1)
        off_diag = anp.ones((N, N)) - anp.eye(N)
        F = off_diag / (T(w_repeated) - w_repeated + anp.eye(N))
        return _dot(v * wg[..., anp.newaxis, :] + _dot(v, F * _dot(T(v), vg)), T(v))
    return vjp
defvjp(eigh, grad_eigh)

def grad_cholesky(L, A):
    # Based on Iain Murray's note http://arxiv.org/abs/1602.07527
    # scipy's dtrtrs wrapper, solve_triangular, doesn't broadcast along leading
    # dimensions, so we just call a generic LU solve instead of directly using
    # backsubstitution (also, we factor twice...)
    solve_trans = lambda a, b: solve(T(a), b)
    phi = lambda X: anp.tril(X) / (1. + anp.eye(X.shape[-1]))
    def conjugate_solve(L, X):
        # X -> L^{-T} X L^{-1}
        return solve_trans(L, T(solve_trans(L, T(X))))

    def vjp(g):
        S = conjugate_solve(L, phi(anp.einsum('...ki,...kj->...ij', L, g)))
        return (S + T(S)) / 2.
    return vjp
defvjp(cholesky, grad_cholesky)

def grad_svd(usv_, a, full_matrices=True, compute_uv=True):
    def vjp(g):
        usv = usv_

        if not compute_uv:
            s = usv

            # Need U and V so do the whole svd anyway...
            usv = svd(a, full_matrices=False)
            u = usv[0]
            v = T(usv[2])

            return _dot(u * g[..., anp.newaxis, :], T(v))

        elif full_matrices:
            raise NotImplementedError(
                "Gradient of svd not implemented for full_matrices=True")

        else:
            u = usv[0]
            s = usv[1]
            v = T(usv[2])

            m, n = a.shape[-2:]

            k = anp.min((m, n))
            # broadcastable identity array with shape (1, 1, ..., 1, k, k)
            i = anp.reshape(anp.eye(k), anp.concatenate((anp.ones(a.ndim - 2, dtype=int), (k, k))))

            f = 1 / (s[..., anp.newaxis, :]**2 - s[..., :, anp.newaxis]**2 + i)

            if m < n:
                gu = g[0]
                gs = g[1]
                gv = T(g[2])

                utgu = _dot(T(u), gu)
                vtgv = _dot(T(v), gv)

                i_minus_vvt = (anp.reshape(anp.eye(n), anp.concatenate((anp.ones(a.ndim - 2, dtype=int), (n, n)))) -
                                _dot(v, T(v)))

                t1 = (f * (utgu - T(utgu))) * s[..., anp.newaxis, :]
                t1 = t1 + i * gs[..., :, anp.newaxis]
                t1 = t1 + s[..., :, anp.newaxis] * (f * (vtgv - T(vtgv)))

                t1 = _dot(_dot(u, t1), T(v))

                t1 = t1 + _dot(_dot(u / s[..., anp.newaxis, :], T(gv)), i_minus_vvt)

                return t1

            elif m == n:
                gu = g[0]
                gs = g[1]
                gv = T(g[2])

                utgu = _dot(T(u), gu)
                vtgv = _dot(T(v), gv)

                t1 = (f * (utgu - T(utgu))) * s[..., anp.newaxis, :]
                t1 = t1 + i * gs[..., :, anp.newaxis]
                t1 = t1 + s[..., :, anp.newaxis] * (f * (vtgv - T(vtgv)))

                t1 = _dot(_dot(u, t1), T(v))

                return t1

            elif m > n:
                gu = g[0]
                gs = g[1]
                gv = T(g[2])

                utgu = _dot(T(u), gu)
                vtgv = _dot(T(v), gv)

                i_minus_uut = (anp.reshape(anp.eye(m), anp.concatenate((anp.ones(a.ndim - 2, dtype=int), (m, m)))) -
                                _dot(u, T(u)))

                t1 = (f * (utgu - T(utgu))) * s[..., anp.newaxis, :]
                t1 = t1 + i * gs[..., :, anp.newaxis]
                t1 = t1 + s[..., :, anp.newaxis] * (f * (vtgv - T(vtgv)))

                t1 = _dot(_dot(u, t1), T(v))

                t1 = t1 + _dot(i_minus_uut, _dot(gu, T(v) / s[..., :, anp.newaxis]))

                return t1
    return vjp
defvjp(svd, grad_svd)
