from __future__ import absolute_import
import itertools
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.linalg as spla
from autograd.util import *
from autograd import grad
from builtins import range
from functools import partial

npr.seed(1)

def check_symmetric_matrix_grads(fun, *args):
    def symmetrize(A):
        L = np.tril(A)
        return (L + T(L))/2.
    new_fun = lambda *args: fun(symmetrize(args[0]), *args[1:])
    return check_grads(new_fun, *args)

T = lambda A : np.swapaxes(A, -1, -2)

def rand_psd(D):
    mat = npr.randn(D,D)
    return np.dot(mat, mat.T)

def test_inv():
    def fun(x): return to_scalar(np.linalg.inv(x))
    d_fun = lambda x : to_scalar(grad(fun)(x))
    D = 8
    mat = npr.randn(D, D)
    mat = np.dot(mat, mat) + 1.0 * np.eye(D)
    check_grads(fun, mat)
    check_grads(d_fun, mat)

def test_inv_3d():
    fun = lambda x: to_scalar(np.linalg.inv(x))
    d_fun = lambda x : to_scalar(grad(fun)(x))

    D = 4
    mat = npr.randn(D, D, D) + 5*np.eye(D)
    check_grads(fun, mat)
    check_grads(d_fun, mat)

    mat = npr.randn(D, D, D, D) + 5*np.eye(D)
    check_grads(fun, mat)
    check_grads(d_fun, mat)

def test_solve_arg1():
    D = 8
    A = npr.randn(D, D) + 10.0 * np.eye(D)
    B = npr.randn(D, D - 1)
    def fun(a): return to_scalar(np.linalg.solve(a, B))
    d_fun = lambda x : to_scalar(grad(fun)(x))
    check_grads(fun, A)
    check_grads(d_fun, A)

def test_solve_arg1_1d():
    D = 8
    A = npr.randn(D, D) + 10.0 * np.eye(D)
    B = npr.randn(D)
    def fun(a): return to_scalar(np.linalg.solve(a, B))
    d_fun = lambda x : to_scalar(grad(fun)(x))
    check_grads(fun, A)
    check_grads(d_fun, A)

def test_solve_arg2():
    D = 6
    A = npr.randn(D, D) + 1.0 * np.eye(D)
    B = npr.randn(D, D - 1)
    def fun(b): return to_scalar(np.linalg.solve(A, b))
    d_fun = lambda x : to_scalar(grad(fun)(x))
    check_grads(fun, B)
    check_grads(d_fun, B)

def test_solve_arg1_3d():
    D = 4
    A = npr.randn(D+1, D, D) + 5*np.eye(D)
    B = npr.randn(D+1, D)
    fun = lambda A: to_scalar(np.linalg.solve(A, B))
    d_fun = lambda A: to_scalar(grad(fun)(A))
    check_grads(fun, A)
    check_grads(d_fun, A)

def test_solve_arg1_3d_3d():
    D = 4
    A = npr.randn(D+1, D, D) + 5*np.eye(D)
    B = npr.randn(D+1, D, D+2)
    fun = lambda A: to_scalar(np.linalg.solve(A, B))
    d_fun = lambda A: to_scalar(grad(fun)(A))
    check_grads(fun, A)
    check_grads(d_fun, A)

def test_det():
    def fun(x): return np.linalg.det(x)
    d_fun = lambda x : to_scalar(grad(fun)(x))
    D = 6
    mat = npr.randn(D, D)
    check_grads(fun, mat)
    check_grads(d_fun, mat)

def test_det_3d():
    fun = lambda x: to_scalar(np.linalg.det(x))
    d_fun = lambda x: to_scalar(grad(fun)(x))
    D = 3
    mat = npr.randn(D, D, D)
    check_grads(fun, mat)
    check_grads(d_fun, mat)

def test_slogdet():
    def fun(x):
        sign, logdet = np.linalg.slogdet(x)
        return logdet
    d_fun = lambda x : to_scalar(grad(fun)(x))
    D = 6
    mat = npr.randn(D, D)
    check_grads(fun, mat)
    check_grads(fun, -mat)
    check_grads(d_fun, mat)

def test_slogdet_3d():
    fun = lambda x: np.sum(np.linalg.slogdet(x)[1])
    d_fun = lambda x: to_scalar(grad(fun)(x))
    mat = np.concatenate([(rand_psd(5) + 5*np.eye(5))[None,...] for _ in range(3)])
    check_grads(fun, mat)
    check_grads(d_fun, mat)

def test_vector_2norm():
    def fun(x): return to_scalar(np.linalg.norm(x))
    d_fun = lambda x: to_scalar(grad(fun)(x))
    D = 6
    vec = npr.randn(D)
    check_grads(fun, vec)
    check_grads(d_fun, vec)

def test_frobenius_norm():
    def fun(x): return to_scalar(np.linalg.norm(x))
    d_fun = lambda x : to_scalar(grad(fun)(x))
    D = 6
    mat = npr.randn(D, D-1)
    check_grads(fun, mat)
    check_grads(d_fun, mat)

def test_frobenius_norm_axis():
    def fun(x): return to_scalar(np.linalg.norm(x, axis=(0, 1)))
    d_fun = lambda x : to_scalar(grad(fun)(x))
    D = 6
    mat = npr.randn(D, D-1, D-2)
    check_grads(fun, mat)
    check_grads(d_fun, mat)

def test_vector_norm_ord():
    def helper(size, ord):
        def fun(x): return np.linalg.norm(x, ord=ord)
        vec = npr.randn(size)
        check_grads(fun, vec)
    for ord in range(2,5):
        yield helper, 6, ord

def test_norm_axis():
    def helper(shape, axis):
        def fun(x): return to_scalar(np.linalg.norm(x, axis=axis))
        arr = npr.randn(*shape)
        check_grads(fun, arr)
    for axis in range(3):
        yield helper, (6,5,4), axis

def test_norm_nuclear():
    def fun(x): return to_scalar(np.linalg.norm(x, ord='nuc'))
    d_fun = lambda x : to_scalar(grad(fun)(x))
    D = 6
    mat = npr.randn(D, D-1)
    check_grads(fun, mat)
    check_grads(d_fun, mat)

def test_norm_nuclear_axis():
    def fun(x): return to_scalar(np.linalg.norm(x, ord='nuc', axis=(0, 1)))
    d_fun = lambda x : to_scalar(grad(fun)(x))
    D = 6
    mat = npr.randn(D, D-1, D-2)
    check_grads(fun, mat)
    check_grads(d_fun, mat)

def test_eigvalh_lower():
    def fun(x):
        w, v = np.linalg.eigh(x)
        return to_scalar(w) + to_scalar(v)
    d_fun = lambda x : to_scalar(grad(fun)(x))
    D = 6
    mat = npr.randn(D, D)
    hmat = np.dot(mat.T, mat)
    check_symmetric_matrix_grads(fun, hmat)
    check_symmetric_matrix_grads(d_fun, hmat)

def test_eigvalh_upper():
    def fun(x):
        w, v = np.linalg.eigh(x, 'U')
        return to_scalar(w) + to_scalar(v)
    d_fun = lambda x : to_scalar(grad(fun)(x))
    D = 6
    mat = npr.randn(D, D)
    hmat = np.dot(mat.T, mat)
    check_symmetric_matrix_grads(fun, hmat)
    check_symmetric_matrix_grads(d_fun, hmat)

broadcast_dot_transpose = partial(np.einsum, '...ij,...kj->...ik')
def test_eigvalh_lower_broadcasting():
    def fun(x):
        w, v = np.linalg.eigh(x)
        return to_scalar(w) + to_scalar(v)
    d_fun = lambda x : to_scalar(grad(fun)(x))
    D = 6
    mat = npr.randn(2, 3, D, D) + 10 * np.eye(D)[None,None,...]
    hmat = broadcast_dot_transpose(mat, mat)
    check_symmetric_matrix_grads(fun, hmat)
    check_symmetric_matrix_grads(d_fun, hmat)

def test_eigvalh_upper_broadcasting():
    def fun(x):
        w, v = np.linalg.eigh(x, 'U')
        return to_scalar(w) + to_scalar(v)
    d_fun = lambda x : to_scalar(grad(fun)(x))
    D = 6
    mat = npr.randn(2, 3, D, D) + 10 * np.eye(D)[None,None,...]
    hmat = broadcast_dot_transpose(mat, mat)
    check_symmetric_matrix_grads(fun, hmat)
    check_symmetric_matrix_grads(d_fun, hmat)

def test_cholesky():
    fun = lambda A: to_scalar(np.linalg.cholesky(A))
    fun2 = lambda A: to_scalar(grad(fun)(A))
    check_symmetric_matrix_grads(fun, rand_psd(6))
    check_symmetric_matrix_grads(fun2, rand_psd(6))

def test_cholesky_broadcast():
    fun = lambda A: to_scalar(np.linalg.cholesky(A))
    fun2 = lambda A: to_scalar(grad(fun)(A))
    A = np.concatenate([rand_psd(6)[None, :, :] for i in range(3)], axis=0)
    check_symmetric_matrix_grads(fun, A)
    check_symmetric_matrix_grads(fun2, A)

def test_cholesky_reparameterization_trick():
    def fun(A):
        rng = np.random.RandomState(0)
        z = np.dot(np.linalg.cholesky(A), rng.randn(A.shape[0]))
        return np.linalg.norm(z)
    check_symmetric_matrix_grads(fun, rand_psd(6))

def test_sqrtm():
    def fun(A):
        return to_scalar(spla.sqrtm(A))
    check_symmetric_matrix_grads(fun, rand_psd(6))

def test_solve_triangular_arg1():
    D = 6
    b = npr.randn(D)
    trans_options = ['T', 'N', 'C', 0, 1, 2]
    lower_options = [True, False]
    for trans, lower in itertools.product(trans_options, lower_options):
        def fun(A):
            return to_scalar(spla.solve_triangular(A, b, trans=trans, lower=lower))
        yield check_grads, fun, npr.randn(D, D) + 10*np.eye(D)

def test_solve_triangular_arg2_1d():
    D = 6
    A = npr.randn(D, D) + 10*np.eye(D)
    trans_options = ['T', 'N', 'C', 0, 1, 2]
    lower_options = [True, False]
    for trans, lower in itertools.product(trans_options, lower_options):
        def fun(b):
            return to_scalar(spla.solve_triangular(A, b, trans=trans, lower=lower))
        yield check_grads, fun, npr.randn(D)

def test_solve_triangular_arg2_2d():
    D = 6
    A = npr.randn(D, D) + 10*np.eye(D)
    trans_options = ['T', 'N', 'C', 0, 1, 2]
    lower_options = [True, False]
    for trans, lower in itertools.product(trans_options, lower_options):
        def fun(B):
            return to_scalar(spla.solve_triangular(A, B, trans=trans, lower=lower))
        yield check_grads, fun, npr.randn(D, D-1)

def test_svd_wide_2d():
    def fun(x):
        u, s, v = np.linalg.svd(x, full_matrices=False)
        return to_scalar(u) + to_scalar(s) + to_scalar(v)
    def d_fun(x):
        return to_scalar(grad(fun)(x))
    m = 3
    n = 5
    mat = npr.randn(m, n)
    check_grads(fun, mat)
    check_grads(d_fun, mat)

def test_svd_wide_3d():
    def fun(x):
        u, s, v = np.linalg.svd(x, full_matrices=False)
        return to_scalar(u) + to_scalar(s) + to_scalar(v)
    def d_fun(x):
        return to_scalar(grad(fun)(x))

    k = 4
    m = 3
    n = 5

    mat = npr.randn(k, m, n)
    check_grads(fun, mat)
    check_grads(d_fun, mat)

def test_svd_square_2d():
    def fun(x):
        u, s, v = np.linalg.svd(x, full_matrices=False)
        return to_scalar(u) + to_scalar(s) + to_scalar(v)
    def d_fun(x):
        return to_scalar(grad(fun)(x))
    m = 4
    n = 4
    mat = npr.randn(m, n)
    check_grads(fun, mat)
    check_grads(d_fun, mat)

def test_svd_square_3d():
    def fun(x):
        u, s, v = np.linalg.svd(x, full_matrices=False)
        return to_scalar(u) + to_scalar(s) + to_scalar(v)
    def d_fun(x):
        return to_scalar(grad(fun)(x))

    k = 3
    m = 4
    n = 4

    mat = npr.randn(k, m, n)
    check_grads(fun, mat)
    check_grads(d_fun, mat)

def test_svd_tall_2d():
    def fun(x):
        u, s, v = np.linalg.svd(x, full_matrices=False)
        return to_scalar(u) + to_scalar(s) + to_scalar(v)
    def d_fun(x):
        return to_scalar(grad(fun)(x))
    m = 5
    n = 3
    mat = npr.randn(m, n)
    check_grads(fun, mat)
    check_grads(d_fun, mat)

def test_svd_tall_3d():
    def fun(x):
        u, s, v = np.linalg.svd(x, full_matrices=False)
        return to_scalar(u) + to_scalar(s) + to_scalar(v)
    def d_fun(x):
        return to_scalar(grad(fun)(x))

    k = 4
    m = 5
    n = 3

    mat = npr.randn(k, m, n)
    check_grads(fun, mat)
    check_grads(d_fun, mat)

def test_svd_only_s_2d():
    def fun(x):
        s = np.linalg.svd(x, full_matrices=False, compute_uv=False)
        return to_scalar(s)
    def d_fun(x):
        return to_scalar(grad(fun)(x))

    m = 5
    n = 3
    mat = npr.randn(m, n)
    check_grads(fun, mat)
    check_grads(d_fun, mat)

def test_svd_only_s_3d():
    def fun(x):
        s = np.linalg.svd(x, full_matrices=False, compute_uv=False)
        return to_scalar(s)
    def d_fun(x):
        return to_scalar(grad(fun)(x))

    k = 4
    m = 5
    n = 3

    mat = npr.randn(k, m, n)
    check_grads(fun, mat)
    check_grads(d_fun, mat)
