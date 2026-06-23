from functools import partial

import numpy as onp
import pytest

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import tuple
from autograd.test_util import check_grads

# Fwd mode not yet implemented
check_grads = partial(check_grads, modes=["rev"])


def check_symmetric_matrix_grads(fun, **grad_check_kwargs):
    def check(*args):
        def symmetrize(A):
            L = np.tril(A)
            return (L + T(L)) / 2.0

        new_fun = lambda *args: fun(symmetrize(args[0]), *args[1:])
        check_grads(new_fun, **grad_check_kwargs)(*args)

    return check


T = lambda A: np.swapaxes(A, -1, -2)


def rand_psd(D):
    rng = npr.RandomState(42)
    mat = rng.randn(D, D)
    return np.dot(mat, mat.T)


def test_inv():
    rng = npr.RandomState(42)

    def fun(x):
        return np.linalg.inv(x)

    D = 8
    mat = rng.randn(D, D)
    mat = np.dot(mat, mat) + 1.0 * np.eye(D)
    check_grads(fun)(mat)


def test_pinv():
    def fun(x):
        return np.linalg.pinv(x)

    N = 5
    D = 2
    rs = npr.RandomState(0)
    ## Non-square matrices:
    for M in range(N // 2, N + N // 2 + 1):
        mat = rs.randn(N, M)
        check_grads(fun)(mat)
        # Stacked
        mat = rs.randn(D, N, M)
        check_grads(fun)(mat)

    ## Square, low (fixed) rank matrices
    def fun_low_rank(x):
        return np.linalg.pinv(np.linalg._dot(np.linalg.T(x), x))

    for M in range(N // 2, N + N // 2 + 1):
        mat = rs.randn(N, M)
        check_grads(fun_low_rank)(mat)
        # Stacked
        mat = rs.randn(D, N, M)
        check_grads(fun_low_rank)(mat)


def test_inv_3d():
    rng = npr.RandomState(42)
    fun = lambda x: np.linalg.inv(x)

    D = 4
    mat = rng.randn(D, D, D) + 5 * np.eye(D)
    check_grads(fun)(mat)

    mat = rng.randn(D, D, D, D) + 5 * np.eye(D)
    check_grads(fun)(mat)


def test_solve_arg1():
    rng = npr.RandomState(42)
    D = 8
    A = rng.randn(D, D) + 10.0 * np.eye(D)
    B = rng.randn(D, D - 1)

    def fun(a):
        return np.linalg.solve(a, B)

    check_grads(fun)(A)


def test_solve_arg1_1d():
    rng = npr.RandomState(42)
    D = 8
    A = rng.randn(D, D) + 10.0 * np.eye(D)
    B = rng.randn(D)

    def fun(a):
        return np.linalg.solve(a, B)

    check_grads(fun)(A)


def test_solve_arg2():
    rng = npr.RandomState(42)
    D = 6
    A = rng.randn(D, D) + 1.0 * np.eye(D)
    B = rng.randn(D, D - 1)

    def fun(b):
        return np.linalg.solve(A, b)

    check_grads(fun)(B)


def test_solve_arg1_3d():
    rng = npr.RandomState(42)
    D = 4
    A = rng.randn(D + 1, D, D) + 5 * np.eye(D)
    B = rng.randn(D + 1, D)
    if onp.lib.NumpyVersion(onp.__version__) < "2.0.0":
        fun = lambda A: np.linalg.solve(A, B)
    else:
        fun = lambda A: np.linalg.solve(A, B[..., None])[..., 0]
    check_grads(fun)(A)


def test_solve_arg1_3d_3d():
    rng = npr.RandomState(42)
    D = 4
    A = rng.randn(D + 1, D, D) + 5 * np.eye(D)
    B = rng.randn(D + 1, D, D + 2)
    fun = lambda A: np.linalg.solve(A, B)
    check_grads(fun)(A)


def test_det():
    rng = npr.RandomState(42)

    def fun(x):
        return np.linalg.det(x)

    D = 6
    mat = rng.randn(D, D)
    check_grads(fun)(mat)


def test_det_3d():
    rng = npr.RandomState(42)
    fun = lambda x: np.linalg.det(x)
    D = 3
    mat = rng.randn(D, D, D)
    check_grads(fun)(mat)


def test_slogdet():
    rng = npr.RandomState(42)

    def fun(x):
        sign, logdet = np.linalg.slogdet(x)
        return logdet

    D = 6
    mat = rng.randn(D, D)
    check_grads(fun)(mat)
    check_grads(fun)(-mat)


def test_slogdet_3d():
    fun = lambda x: np.sum(np.linalg.slogdet(x)[1])
    mat = np.concatenate([(rand_psd(5) + 5 * np.eye(5))[None, ...] for _ in range(3)])
    check_grads(fun)(mat)


def test_vector_2norm():
    rng = npr.RandomState(42)

    def fun(x):
        return np.linalg.norm(x)

    D = 6
    vec = rng.randn(D)
    check_grads(fun, modes=["fwd", "rev"])(vec)


def test_norm_list_of_boxes():
    from autograd import grad

    # d/dx sqrt(x**2 + (x + 2)**2) at x=0 is 2/sqrt(4) = 1.
    assert grad(lambda x: np.linalg.norm([x, x + 2]))(0.0) == 1.0
    assert grad(lambda x: np.linalg.norm((x, x + 2)))(0.0) == 1.0


def test_vector_2norm_complex():
    rng = npr.RandomState(42)

    def fun(x):
        return np.linalg.norm(x)

    D = 6
    vec = rng.randn(D) + 1j * rng.randn(D)
    check_grads(fun)(vec)


def test_frobenius_norm():
    rng = npr.RandomState(42)

    def fun(x):
        return np.linalg.norm(x)

    D = 6
    mat = rng.randn(D, D - 1)
    check_grads(fun, modes=["fwd", "rev"])(mat)


def test_frobenius_norm_complex():
    rng = npr.RandomState(42)

    def fun(x):
        return np.linalg.norm(x)

    D = 6
    mat = rng.randn(D, D - 1) + 1j * rng.randn(D, D - 1)
    check_grads(fun)(mat)


def test_frobenius_norm_axis():
    rng = npr.RandomState(42)

    def fun(x):
        return np.linalg.norm(x, axis=(0, 1))

    D = 6
    mat = rng.randn(D, D - 1, D - 2)
    check_grads(fun, modes=["fwd", "rev"])(mat)


def test_frobenius_norm_axis_complex():
    rng = npr.RandomState(42)

    def fun(x):
        return np.linalg.norm(x, axis=(0, 1))

    D = 6
    mat = rng.randn(D, D - 1, D - 2) + 1j * rng.randn(D, D - 1, D - 2)
    check_grads(fun)(mat)


@pytest.mark.parametrize("ord", range(2, 5))
@pytest.mark.parametrize("size", [6])
def test_vector_norm_ord(size, ord):
    rng = npr.RandomState(42)

    def fun(x):
        return np.linalg.norm(x, ord=ord)

    vec = rng.randn(size)
    check_grads(fun, modes=["fwd", "rev"])(vec)


@pytest.mark.parametrize("ord", range(2, 5))
@pytest.mark.parametrize("size", [6])
def test_vector_norm_ord_complex(size, ord):
    rng = npr.RandomState(42)

    def fun(x):
        return np.linalg.norm(x, ord=ord)

    vec = rng.randn(size) + 1j * rng.randn(size)
    check_grads(fun)(vec)


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("shape", [(6, 5, 4)])
def test_norm_axis(shape, axis):
    rng = npr.RandomState(42)

    def fun(x):
        return np.linalg.norm(x, axis=axis)

    arr = rng.randn(*shape)
    check_grads(fun, modes=["fwd", "rev"])(arr)


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("shape", [(6, 5, 4)])
def test_norm_axis_complex(shape, axis):
    rng = npr.RandomState(42)

    def fun(x):
        return np.linalg.norm(x, axis=axis)

    arr = rng.randn(*shape) + 1j * rng.randn(*shape)
    check_grads(fun)(arr)


def test_norm_nuclear():
    rng = npr.RandomState(42)

    def fun(x):
        return np.linalg.norm(x, ord="nuc")

    D = 6
    mat = rng.randn(D, D - 1)
    # Order 1 because the jvp of the svd is not implemented
    check_grads(fun, modes=["fwd", "rev"], order=1)(mat)


def test_norm_nuclear_complex():
    rng = npr.RandomState(42)

    def fun(x):
        return np.linalg.norm(x, ord="nuc")

    D = 6
    mat = rng.randn(D, D - 1) + 1j * rng.randn(D, D - 1)
    check_grads(fun)(mat)


def test_norm_nuclear_axis():
    rng = npr.RandomState(42)

    def fun(x):
        return np.linalg.norm(x, ord="nuc", axis=(0, 1))

    D = 6
    mat = rng.randn(D, D - 1, D - 2)
    # Order 1 because the jvp of the svd is not implemented
    check_grads(fun, modes=["fwd", "rev"], order=1)(mat)


def test_norm_nuclear_axis_complex():
    rng = npr.RandomState(42)

    def fun(x):
        return np.linalg.norm(x, ord="nuc", axis=(0, 1))

    D = 6
    mat = rng.randn(D, D - 1, D - 2) + 1j * rng.randn(D, D - 1, D - 2)
    check_grads(fun)(mat)


def test_eigvalh_lower():
    rng = npr.RandomState(42)

    def fun(x):
        w, v = np.linalg.eigh(x)
        return tuple((w, v))

    D = 6
    mat = rng.randn(D, D)
    check_grads(fun)(mat)


def test_eigvalh_upper():
    rng = npr.RandomState(42)

    def fun(x):
        w, v = np.linalg.eigh(x, "U")
        return tuple((w, v))

    D = 6
    mat = rng.randn(D, D)
    check_grads(fun)(mat)


broadcast_dot_transpose = partial(np.einsum, "...ij,...kj->...ik")


def test_eigvalh_lower_broadcasting():
    rng = npr.RandomState(42)

    def fun(x):
        w, v = np.linalg.eigh(x)
        return tuple((w, v))

    D = 6
    mat = rng.randn(2, 3, D, D) + 10 * np.eye(D)[None, None, ...]
    hmat = broadcast_dot_transpose(mat, mat)
    check_grads(fun)(hmat)


def test_eigvalh_upper_broadcasting():
    rng = npr.RandomState(42)

    def fun(x):
        w, v = np.linalg.eigh(x, "U")
        return tuple((w, v))

    D = 6
    mat = rng.randn(2, 3, D, D) + 10 * np.eye(D)[None, None, ...]
    hmat = broadcast_dot_transpose(mat, mat)
    check_grads(fun)(hmat)


# For complex-valued matrices, the eigenvectors could have arbitrary phases (gauge)
# which makes it impossible to compare to numerical derivatives. So we take the
# absolute value to get rid of that phase.
def test_eigvalh_lower_complex():
    rng = npr.RandomState(42)

    def fun(x):
        w, v = np.linalg.eigh(x)
        return tuple((w, np.abs(v)))

    D = 6
    mat = rng.randn(D, D) + 1j * rng.randn(D, D)
    check_grads(fun)(mat)


def test_eigvalh_upper_complex():
    rng = npr.RandomState(42)

    def fun(x):
        w, v = np.linalg.eigh(x, "U")
        return tuple((w, np.abs(v)))

    D = 6
    mat = rng.randn(D, D) + 1j * rng.randn(D, D)
    check_grads(fun)(mat)


# Note eigenvalues and eigenvectors for real matrix can still be complex
def test_eig_real():
    rng = npr.RandomState(42)

    def fun(x):
        w, v = np.linalg.eig(x)
        return tuple((np.abs(w), np.abs(v)))

    D = 8
    mat = rng.randn(D, D)
    check_grads(fun)(mat)


def test_eig_complex():
    rng = npr.RandomState(42)

    def fun(x):
        w, v = np.linalg.eig(x)
        return tuple((w, np.abs(v)))

    D = 8
    mat = rng.randn(D, D) + 1.0j * rng.randn(D, D)
    check_grads(fun)(mat)


def test_eig_batched():
    rng = npr.RandomState(42)

    def fun(x):
        w, v = np.linalg.eig(x)
        return tuple((w, np.abs(v)))

    D = 8
    b = 5
    mat = rng.randn(b, D, D) + 1.0j * rng.randn(b, D, D)
    check_grads(fun)(mat)


def test_cholesky():
    fun = lambda A: np.linalg.cholesky(A)
    check_symmetric_matrix_grads(fun)(rand_psd(6))


def test_cholesky_broadcast():
    fun = lambda A: np.linalg.cholesky(A)
    A = np.concatenate([rand_psd(6)[None, :, :] for i in range(3)], axis=0)
    check_symmetric_matrix_grads(fun)(A)


def test_cholesky_reparameterization_trick():
    def fun(A):
        rng = np.random.RandomState(0)
        z = np.dot(np.linalg.cholesky(A), rng.randn(A.shape[0]))
        return np.linalg.norm(z)

    check_symmetric_matrix_grads(fun)(rand_psd(6))


def test_svd_wide_2d():
    rng = npr.RandomState(42)

    def fun(x):
        u, s, v = np.linalg.svd(x, full_matrices=False)
        return tuple((u, s, v))

    m = 3
    n = 5
    mat = rng.randn(m, n)
    check_grads(fun)(mat)


def test_svd_wide_2d_complex():
    rng = npr.RandomState(42)

    def fun(x):
        u, s, v = np.linalg.svd(x, full_matrices=False)
        return tuple((np.abs(u), s, np.abs(v)))

    m = 3
    n = 5
    mat = rng.randn(m, n) + 1j * rng.randn(m, n)
    check_grads(fun)(mat)


def test_svd_wide_3d():
    rng = npr.RandomState(42)

    def fun(x):
        u, s, v = np.linalg.svd(x, full_matrices=False)
        return tuple((u, s, v))

    k = 4
    m = 3
    n = 5
    mat = rng.randn(k, m, n)
    check_grads(fun)(mat)


def test_svd_wide_3d_complex():
    rng = npr.RandomState(42)

    def fun(x):
        u, s, v = np.linalg.svd(x, full_matrices=False)
        return tuple((np.abs(u), s, np.abs(v)))

    k = 4
    m = 3
    n = 5
    mat = rng.randn(k, m, n) + 1j * rng.randn(k, m, n)
    check_grads(fun)(mat)


def test_svd_square_2d():
    rng = npr.RandomState(42)

    def fun(x):
        u, s, v = np.linalg.svd(x, full_matrices=False)
        return tuple((u, s, v))

    m = 4
    n = 4
    mat = rng.randn(m, n)
    check_grads(fun)(mat)


def test_svd_square_2d_complex():
    rng = npr.RandomState(42)

    def fun(x):
        u, s, v = np.linalg.svd(x, full_matrices=False)
        return tuple((np.abs(u), s, np.abs(v)))

    m = 4
    n = 4
    mat = rng.randn(m, n) + 1j * rng.randn(m, n)
    check_grads(fun)(mat)


def test_svd_square_3d():
    rng = npr.RandomState(42)

    def fun(x):
        u, s, v = np.linalg.svd(x, full_matrices=False)
        return tuple((u, s, v))

    k = 3
    m = 4
    n = 4
    mat = rng.randn(k, m, n)
    check_grads(fun)(mat)


def test_svd_square_3d_complex():
    rng = npr.RandomState(42)

    def fun(x):
        u, s, v = np.linalg.svd(x, full_matrices=False)
        return tuple((np.abs(u), s, np.abs(v)))

    k = 3
    m = 4
    n = 4
    mat = rng.randn(k, m, n) + 1j * rng.randn(k, m, n)
    check_grads(fun)(mat)


def test_svd_tall_2d():
    rng = npr.RandomState(42)

    def fun(x):
        u, s, v = np.linalg.svd(x, full_matrices=False)
        return tuple((u, s, v))

    m = 5
    n = 3
    mat = rng.randn(m, n)
    check_grads(fun)(mat)


def test_svd_tall_2d_complex():
    rng = npr.RandomState(42)

    def fun(x):
        u, s, v = np.linalg.svd(x, full_matrices=False)
        return tuple((np.abs(u), s, np.abs(v)))

    m = 5
    n = 3
    mat = rng.randn(m, n) + 1j * rng.randn(m, n)
    check_grads(fun)(mat)


def test_svd_tall_3d():
    rng = npr.RandomState(42)

    def fun(x):
        u, s, v = np.linalg.svd(x, full_matrices=False)
        return tuple((u, s, v))

    k = 4
    m = 5
    n = 3
    mat = rng.randn(k, m, n)
    check_grads(fun)(mat)


def test_svd_tall_3d_complex():
    rng = npr.RandomState(42)

    def fun(x):
        u, s, v = np.linalg.svd(x, full_matrices=False)
        return tuple((np.abs(u), s, np.abs(v)))

    k = 4
    m = 5
    n = 3
    mat = rng.randn(k, m, n) + 1j * rng.randn(k, m, n)
    check_grads(fun)(mat)


def test_svd_only_s_2d():
    rng = npr.RandomState(42)

    def fun(x):
        s = np.linalg.svd(x, full_matrices=False, compute_uv=False)
        return s

    m = 5
    n = 3
    mat = rng.randn(m, n)
    check_grads(fun)(mat)


def test_svd_only_s_2d_complex():
    rng = npr.RandomState(42)

    def fun(x):
        s = np.linalg.svd(x, full_matrices=False, compute_uv=False)
        return s

    m = 5
    n = 3
    mat = rng.randn(m, n) + 1j * rng.randn(m, n)
    check_grads(fun)(mat)


def test_svd_only_s_3d():
    rng = npr.RandomState(42)

    def fun(x):
        s = np.linalg.svd(x, full_matrices=False, compute_uv=False)
        return s

    k = 4
    m = 5
    n = 3
    mat = rng.randn(k, m, n)
    check_grads(fun)(mat)


def test_svd_only_s_3d_complex():
    rng = npr.RandomState(42)

    def fun(x):
        s = np.linalg.svd(x, full_matrices=False, compute_uv=False)
        return s

    k = 4
    m = 5
    n = 3
    mat = rng.randn(k, m, n) + 1j * rng.randn(k, m, n)
    check_grads(fun)(mat)
