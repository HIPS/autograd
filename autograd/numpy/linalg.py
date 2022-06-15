from __future__ import absolute_import
from functools import partial
import numpy.linalg as npla
from .numpy_wrapper import wrap_namespace
from . import numpy_wrapper as anp
from autograd.extend import defvjp, defjvp

wrap_namespace(npla.__dict__, globals())

# Some formulas are from
# "An extended collection of matrix derivative results
#  for forward and reverse mode algorithmic differentiation"
# by Mike Giles
# https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf

# transpose by swapping last two dimensions
def T(x): return anp.swapaxes(x, -1, -2)

_dot = partial(anp.einsum, '...ij,...jk->...ik')

# batched diag
_diag = lambda a: anp.eye(a.shape[-1])*a

# batched diagonal, similar to matrix_diag in tensorflow
def _matrix_diag(a):
    reps = anp.array(a.shape)
    reps[:-1] = 1
    reps[-1] = a.shape[-1]
    newshape = list(a.shape) + [a.shape[-1]]
    return _diag(anp.tile(a, reps).reshape(newshape))

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

def norm_vjp(ans, x, ord=None, axis=None):
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

    check_implemented()
    def vjp(g):
        if ord in (None, 2, 'fro'):
            return expand(g / ans) * x
        elif ord == 'nuc':
            x_rolled = roll(x)
            u, s, vt = svd(x_rolled, full_matrices=False)
            uvt_rolled = _dot(u, vt)
            # Roll the matrix axes back to their correct positions
            uvt = unroll(uvt_rolled)
            g = expand(g)
            return g * uvt
        else:
            # see https://en.wikipedia.org/wiki/Norm_(mathematics)#p-norm
            return expand(g / ans**(ord-1)) * x * anp.abs(x)**(ord-2)
    return vjp
defvjp(norm, norm_vjp)

def norm_jvp(g, ans, x, ord=None, axis=None):
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
        contract = lambda a: anp.sum(a)
    else:
        contract = partial(anp.sum, axis=axis)

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

    check_implemented()
    if ord in (None, 2, 'fro'):
        return contract(g * x) / ans
    elif ord == 'nuc':
        x_rolled = roll(x)
        u, s, vt = svd(x_rolled, full_matrices=False)
        uvt_rolled = _dot(u, vt)
        # Roll the matrix axes back to their correct positions
        uvt = unroll(uvt_rolled)
        return contract(g * uvt)
    else:
        # see https://en.wikipedia.org/wiki/Norm_(mathematics)#p-norm
        return contract(g * x * anp.abs(x)**(ord-2)) / ans**(ord-1)
defjvp(norm, norm_jvp)

def grad_eigh(ans, x, UPLO='L'):
    """Gradient for eigenvalues and vectors of a symmetric matrix."""
    N = x.shape[-1]
    w, v = ans              # Eigenvalues, eigenvectors.
    vc = anp.conj(v)

    def vjp(g):
        wg, vg = g          # Gradient w.r.t. eigenvalues, eigenvectors.
        w_repeated = anp.repeat(w[..., anp.newaxis], N, axis=-1)

        # Eigenvalue part
        vjp_temp = _dot(vc * wg[..., anp.newaxis, :], T(v))

        # Add eigenvector part only if non-zero backward signal is present.
        # This can avoid NaN results for degenerate cases if the function depends
        # on the eigenvalues only.
        if anp.any(vg):
            off_diag = anp.ones((N, N)) - anp.eye(N)
            F = off_diag / (T(w_repeated) - w_repeated + anp.eye(N))
            vjp_temp += _dot(_dot(vc, F * _dot(T(v), vg)), T(v))

        # eigh always uses only the lower or the upper part of the matrix
        # we also have to make sure broadcasting works
        reps = anp.array(x.shape)
        reps[-2:] = 1

        if UPLO == 'L':
            tri = anp.tile(anp.tril(anp.ones(N), -1), reps)
        elif UPLO == 'U':
            tri = anp.tile(anp.triu(anp.ones(N), 1), reps)

        return anp.real(vjp_temp)*anp.eye(vjp_temp.shape[-1]) + \
            (vjp_temp + anp.conj(T(vjp_temp))) * tri

    return vjp
defvjp(eigh, grad_eigh)

# https://arxiv.org/pdf/1701.00392.pdf Eq(4.77)
# Note the formula from Sec3.1 in https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf is incomplete
def grad_eig(ans, x):
    """Gradient of a general square (complex valued) matrix"""
    e, u = ans # eigenvalues as 1d array, eigenvectors in columns
    n = e.shape[-1]
    def vjp(g):
        ge, gu = g
        ge = _matrix_diag(ge)
        f = 1/(e[..., anp.newaxis, :] - e[..., :, anp.newaxis] + 1.e-20)
        f -= _diag(f)
        ut = anp.swapaxes(u, -1, -2)
        r1 = f * _dot(ut, gu)
        r2 = -f * (_dot(_dot(ut, anp.conj(u)), anp.real(_dot(ut,gu)) * anp.eye(n)))
        r = _dot(_dot(inv(ut), ge + r1 + r2), ut)
        if not anp.iscomplexobj(x):
            r = anp.real(r)
            # the derivative is still complex for real input (imaginary delta is allowed), real output
            # but the derivative should be real in real input case when imaginary delta is forbidden
        return r
    return vjp
defvjp(eig, grad_eig)

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

# https://j-towns.github.io/papers/svd-derivative.pdf
# https://arxiv.org/abs/1909.02659
def grad_svd(usv_, a, full_matrices=True, compute_uv=True):
    def vjp(g):
        usv = usv_

        if not compute_uv:
            s = usv

            # Need U and V so do the whole svd anyway...
            usv = svd(a, full_matrices=False)
            u = usv[0]
            v = anp.conj(T(usv[2]))

            return _dot(anp.conj(u) * g[..., anp.newaxis, :], T(v))

        elif full_matrices:
            raise NotImplementedError(
                "Gradient of svd not implemented for full_matrices=True")

        else:
            u = usv[0]
            s = usv[1]
            v = anp.conj(T(usv[2]))

            m, n = a.shape[-2:]

            k = anp.min((m, n))
            # broadcastable identity array with shape (1, 1, ..., 1, k, k)
            i = anp.reshape(anp.eye(k), anp.concatenate((anp.ones(a.ndim - 2, dtype=int), (k, k))))

            f = 1 / (s[..., anp.newaxis, :]**2 - s[..., :, anp.newaxis]**2 + i)

            gu = g[0]
            gs = g[1]
            gv = anp.conj(T(g[2]))

            utgu = _dot(T(u), gu)
            vtgv = _dot(T(v), gv)
            t1 = (f * (utgu - anp.conj(T(utgu)))) * s[..., anp.newaxis, :]
            t1 = t1 + i * gs[..., :, anp.newaxis]
            t1 = t1 + s[..., :, anp.newaxis] * (f * (vtgv - anp.conj(T(vtgv))))

            if anp.iscomplexobj(u):
                t1 = t1 + 1j*anp.imag(_diag(utgu)) / s[..., anp.newaxis, :]

            t1 = _dot(_dot(anp.conj(u), t1), T(v))

            if m < n:
                i_minus_vvt = (anp.reshape(anp.eye(n), anp.concatenate((anp.ones(a.ndim - 2, dtype=int), (n, n)))) -
                                _dot(v, anp.conj(T(v))))
                t1 = t1 + anp.conj(_dot(_dot(u / s[..., anp.newaxis, :], T(gv)), i_minus_vvt))

                return t1

            elif m == n:
                return t1

            elif m > n:
                i_minus_uut = (anp.reshape(anp.eye(m), anp.concatenate((anp.ones(a.ndim - 2, dtype=int), (m, m)))) -
                                _dot(u, anp.conj(T(u))))
                t1 = t1 + T(_dot(_dot(v/s[..., anp.newaxis, :], T(gu)), i_minus_uut) )

                return t1
    return vjp
defvjp(svd, grad_svd)
