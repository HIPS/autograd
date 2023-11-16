from __future__ import division
from functools import partial
import scipy.linalg

import autograd.numpy as anp
from autograd.numpy.numpy_wrapper import wrap_namespace
from autograd.extend import defvjp, defvjp_argnums, defjvp, defjvp_argnums

wrap_namespace(scipy.linalg.__dict__, globals())  # populates module namespace

def _vjp_sqrtm(ans, A, disp=True, blocksize=64):
    assert disp, "sqrtm vjp not implemented for disp=False"
    ans_transp = anp.transpose(ans)
    def vjp(g):
        return anp.real(solve_sylvester(ans_transp, ans_transp, g))
    return vjp
defvjp(sqrtm, _vjp_sqrtm)

def _flip(a, trans):
    if anp.iscomplexobj(a):
        return 'H' if trans in ('N', 0) else 'N'
    else:
        return 'T' if trans in ('N', 0) else 'N'

def grad_solve_triangular(ans, a, b, trans=0, lower=False, **kwargs):
    tri = anp.tril if (lower ^ (_flip(a, trans) == 'N')) else anp.triu
    transpose = lambda x: x if _flip(a, trans) != 'N' else x.T
    al2d = lambda x: x if x.ndim > 1 else x[...,None]
    def vjp(g):
        v = al2d(solve_triangular(a, g, trans=_flip(a, trans), lower=lower))
        return -transpose(tri(anp.dot(v, al2d(ans).T)))
    return vjp

defvjp(solve_triangular,
       grad_solve_triangular,
       lambda ans, a, b, trans=0, lower=False, **kwargs:
       lambda g: solve_triangular(a, g, trans=_flip(a, trans), lower=lower))

def grad_solve_banded(argnum, ans, l_and_u, a, b):

    updim = lambda x: x if x.ndim == a.ndim else x[...,None]

    def transpose_banded(l_and_u, a):

        # Compute the transpose of a banded matrix.
        # The transpose is itself a banded matrix.

        num_rows = a.shape[0]

        shifts = anp.arange(-l_and_u[1], l_and_u[0]+1)

        T_a = anp.roll(a[:1, :], shifts[0])
        for rr in range(1, num_rows):
            T_a = anp.vstack([T_a, anp.flipud(anp.roll(a[rr:rr+1, :], shifts[rr]))])
        T_a = anp.flipud(T_a)

        T_l_and_u = anp.flip(l_and_u)

        return T_l_and_u, T_a

    def banded_dot(l_and_u, uu, vv):

        # Compute tensor product of vectors uu and vv.
        # Tensor product elements are resticted to the bands specified by l_and_u.

        # TODO: replace the brute-force ravel() by smarter dimension handeling of uu and vv

        # main diagonal
        banded_uv = anp.ravel(uu)*anp.ravel(vv)

        # stack below the sub-diagonals
        for rr in range(1, l_and_u[0]+1):
            banded_uv_rr = anp.hstack([anp.ravel(uu)[rr:]*anp.ravel(vv)[:-rr], anp.zeros(rr)])
            banded_uv = anp.vstack([banded_uv, banded_uv_rr])

        # stack above the sup-diagonals
        for rr in range(1, l_and_u[1]+1):
            banded_uv_rr = anp.hstack([anp.zeros(rr), anp.ravel(uu)[:-rr]*anp.ravel(vv)[rr:]])
            banded_uv = anp.vstack([banded_uv_rr, banded_uv])

        return(banded_uv)

    T_l_and_u, T_a = transpose_banded(l_and_u, a)

    if argnum == 1:
        return lambda g: -banded_dot(l_and_u, updim(solve_banded(T_l_and_u, T_a, g)), anp.transpose(updim(ans)))
    elif argnum == 2:
        return lambda g: solve_banded(T_l_and_u, T_a, g)

defvjp(solve_banded,
       partial(grad_solve_banded, 1),
       partial(grad_solve_banded, 2),
       argnums=[1, 2])

def _jvp_sqrtm(dA, ans, A, disp=True, blocksize=64):
    assert disp, "sqrtm jvp not implemented for disp=False"
    return solve_sylvester(ans, ans, dA)
defjvp(sqrtm, _jvp_sqrtm)

def _jvp_sylvester(argnums, dms, ans, args, _):
    a, b, q = args
    if 0 in argnums:
        da = dms[0]
        db = dms[1] if 1 in argnums else 0
    else:
        da = 0
        db = dms[0] if 1 in argnums else 0
    dq = dms[-1] if 2 in argnums else 0
    rhs = dq - anp.dot(da, ans) - anp.dot(ans, db)
    return solve_sylvester(a, b, rhs)
defjvp_argnums(solve_sylvester, _jvp_sylvester)

def _vjp_sylvester(argnums, ans, args, _):
    a, b, q = args
    def vjp(g):
        vjps = []
        q_vjp = solve_sylvester(anp.transpose(a), anp.transpose(b), g)
        if 0 in argnums: vjps.append(-anp.dot(q_vjp, anp.transpose(ans)))
        if 1 in argnums: vjps.append(-anp.dot(anp.transpose(ans), q_vjp))
        if 2 in argnums: vjps.append(q_vjp)
        return tuple(vjps)
    return vjp
defvjp_argnums(solve_sylvester, _vjp_sylvester)
