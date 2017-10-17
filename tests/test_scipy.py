from __future__ import absolute_import
import numpy as npo
import autograd.numpy as anp
from builtins import range
import itertools
from functools import wraps

def symmetrize_matrix_arg(fun, argnum):
    def T(X): return anp.swapaxes(X, -1, -2) if anp.ndim(X) > 1 else X
    def symmetrize(X): return 0.5 * (X + T(X))
    def symmetrized_fun(*args, **kwargs):
        args = list(args)
        args[argnum] = symmetrize(args[argnum])
        return fun(*args, **kwargs)
    return symmetrized_fun

try:
    import scipy
except:
    from warnings import warn
    warn('Skipping scipy tests.')
else:
    import autograd.numpy as np
    import autograd.numpy.random as npr
    import autograd.scipy.misc
    import autograd.scipy.signal
    import autograd.scipy.stats as stats
    import autograd.scipy.stats.multivariate_normal as mvn
    import autograd.scipy.special as special
    import autograd.scipy.linalg as spla
    from autograd import grad

    from numpy_utils import combo_check, check_grads, unary_ufunc_check, to_scalar
    from test_linalg import check_symmetric_matrix_grads, rand_psd

    npr.seed(1)
    R = npr.randn
    U = npr.uniform

    ### Stats ###
    def test_norm_pdf():    combo_check(stats.norm.pdf,    [0,1,2], [R(4)], [R(4)], [R(4)**2 + 1.1])
    def test_norm_cdf():    combo_check(stats.norm.cdf,    [0,1,2], [R(4)], [R(4)], [R(4)**2 + 1.1])
    def test_norm_logpdf(): combo_check(stats.norm.logpdf, [0,1,2], [R(4)], [R(4)], [R(4)**2 + 1.1])
    def test_norm_logcdf(): combo_check(stats.norm.logcdf, [0,1,2], [R(4)], [R(4)], [R(4)**2 + 1.1])

    def test_norm_pdf_broadcast():    combo_check(stats.norm.pdf,    [0,1,2], [R(4,3)], [R(1,3)], [R(4,1)**2 + 1.1])
    def test_norm_cdf_broadcast():    combo_check(stats.norm.cdf,    [0,1,2], [R(4,3)], [R(1,3)], [R(4,1)**2 + 1.1])
    def test_norm_logpdf_broadcast(): combo_check(stats.norm.logpdf, [0,1,2], [R(4,3)], [R(1,3)], [R(4,1)**2 + 1.1])
    def test_norm_logcdf_broadcast(): combo_check(stats.norm.logcdf, [0,1,2], [R(4,3)], [R(1,3)], [R(4,1)**2 + 1.1])

    def test_poisson_cdf():    combo_check(stats.poisson.cdf,    [1], [np.round(R(4)**2)], [R(4)**2])
    def test_poisson_logpmf(): combo_check(stats.poisson.logpmf, [1], [np.round(R(4)**2)], [R(4)**2])
    def test_poisson_pmf():    combo_check(stats.poisson.pmf,    [1], [np.round(R(4)**2)], [R(4)**2])

    def test_poisson_cdf_broadcast():    combo_check(stats.poisson.cdf,    [1], [np.round(R(4, 3)**2)], [R(4, 3)**2])
    def test_poisson_logpmf_broadcast(): combo_check(stats.poisson.logpmf, [1], [np.round(R(4, 3)**2)], [R(4, 3)**2])
    def test_poisson_pmf_broadcast():    combo_check(stats.poisson.pmf,    [1], [np.round(R(4, 3)**2)], [R(4, 3)**2])

    def test_t_pdf():    combo_check(stats.t.pdf,    [0,1,2,3], [R(4)], [R(4)**2 + 2.1], [R(4)], [R(4)**2 + 2.1])
    def test_t_cdf():    combo_check(stats.t.cdf,    [0,2],     [R(4)], [R(4)**2 + 2.1], [R(4)], [R(4)**2 + 2.1])
    def test_t_logpdf(): combo_check(stats.t.logpdf, [0,1,2,3], [R(4)], [R(4)**2 + 2.1], [R(4)], [R(4)**2 + 2.1])
    def test_t_logcdf(): combo_check(stats.t.logcdf, [0,2],     [R(4)], [R(4)**2 + 2.1], [R(4)], [R(4)**2 + 2.1])

    def test_t_pdf_broadcast():    combo_check(stats.t.pdf,    [0,1,2,3], [R(4,3)], [R(1,3)**2 + 2.1], [R(4,3)], [R(4,1)**2 + 2.1])
    def test_t_cdf_broadcast():    combo_check(stats.t.cdf,    [0,2],     [R(4,3)], [R(1,3)**2 + 2.1], [R(4,3)], [R(4,1)**2 + 2.1])
    def test_t_logpdf_broadcast(): combo_check(stats.t.logpdf, [0,1,2,3], [R(4,3)], [R(1,3)**2 + 2.1], [R(4,3)], [R(4,1)**2 + 2.1])
    def test_t_logcdf_broadcast(): combo_check(stats.t.logcdf, [0,2],     [R(4,3)], [R(1,3)**2 + 2.1], [R(4,3)], [R(4,1)**2 + 2.1])

    def make_psd(mat): return np.dot(mat.T, mat) + np.eye(mat.shape[0])
    def test_mvn_pdf():    combo_check(symmetrize_matrix_arg(mvn.pdf, 2), [0, 1, 2], [R(4)], [R(4)], [make_psd(R(4, 4))], allow_singular=[False])
    def test_mvn_logpdf(): combo_check(symmetrize_matrix_arg(mvn.logpdf, 2), [0, 1, 2], [R(4)], [R(4)], [make_psd(R(4, 4))], allow_singular=[False])
    def test_mvn_entropy():combo_check(mvn.entropy,[0, 1],            [R(4)], [make_psd(R(4, 4))])

    C = np.zeros((4, 4))
    C[0, 0] = C[1, 1] = 1
    # C += 1e-3 * np.eye(4)
    def test_mvn_pdf_sing_cov(): combo_check(mvn.pdf, [0, 1], [np.concatenate((R(2), np.zeros(2)))], [np.concatenate((R(2), np.zeros(2)))], [C], [True])
    def test_mvn_logpdf_sing_cov(): combo_check(mvn.logpdf, [0, 1], [np.concatenate((R(2), np.zeros(2)))], [np.concatenate((R(2), np.zeros(2)))], [C], [True])

    def test_mvn_pdf_broadcast():    combo_check(symmetrize_matrix_arg(mvn.pdf, 2), [0, 1, 2], [R(5, 4)], [R(4)], [make_psd(R(4, 4))])
    def test_mvn_logpdf_broadcast(): combo_check(symmetrize_matrix_arg(mvn.logpdf, 2), [0, 1, 2], [R(5, 4)], [R(4)], [make_psd(R(4, 4))])

    alpha = npr.random(4)**2 + 1.2
    x = stats.dirichlet.rvs(alpha, size=1)[0,:]

    # Need to normalize input so that x's sum to one even when we perturb them to compute numeric gradient.
    def normalize(x): return x / sum(x)
    def normalized_dirichlet_pdf(   x, alpha): return stats.dirichlet.pdf(   normalize(x), alpha)
    def normalized_dirichlet_logpdf(x, alpha): return stats.dirichlet.logpdf(normalize(x), alpha)

    def test_dirichlet_pdf_x():        combo_check(normalized_dirichlet_pdf,    [0], [x], [alpha])
    def test_dirichlet_pdf_alpha():    combo_check(stats.dirichlet.pdf,         [1], [x], [alpha])
    def test_dirichlet_logpdf_x():     combo_check(normalized_dirichlet_logpdf, [0], [x], [alpha])
    def test_dirichlet_logpdf_alpha(): combo_check(stats.dirichlet.logpdf,      [1], [x], [alpha])

    ### Misc ###
    def test_logsumexp1(): combo_check(autograd.scipy.misc.logsumexp, [0], [1.1, R(4), R(3,4)],                axis=[None, 0],    keepdims=[True, False])
    def test_logsumexp2(): combo_check(autograd.scipy.misc.logsumexp, [0], [R(3,4), R(4,5,6), R(1,5)],         axis=[None, 0, 1], keepdims=[True, False])
    def test_logsumexp3(): combo_check(autograd.scipy.misc.logsumexp, [0], [R(4)], b = [np.exp(R(4))],         axis=[None, 0],    keepdims=[True, False])
    def test_logsumexp4(): combo_check(autograd.scipy.misc.logsumexp, [0], [R(3,4),], b = [np.exp(R(3,4))],    axis=[None, 0, 1], keepdims=[True, False])
    def test_logsumexp5(): combo_check(autograd.scipy.misc.logsumexp, [0], [R(2,3,4)], b = [np.exp(R(2,3,4))], axis=[None, 0, 1], keepdims=[True, False])
    def test_logsumexp6():
        x = npr.randn(1,5)
        def f(a): return autograd.scipy.misc.logsumexp(a, axis=1, keepdims=True)
        check_grads(f, x)
        check_grads(lambda a: to_scalar(grad(f)(a)), x)

    ### Signal ###
    def test_convolve_generalization():
        from scipy.signal import convolve as sp_convolve

        ag_convolve = autograd.scipy.signal.convolve
        A_35 = R(3, 5)
        A_34 = R(3, 4)
        A_342 = R(3, 4, 2)
        A_2543 = R(2, 5, 4, 3)
        A_24232 = R(2, 4, 2, 3, 2)

        for mode in ['valid', 'full']:
            assert npo.allclose(ag_convolve(A_35,      A_34, axes=([1], [0]), mode=mode)[1, 2],
                                sp_convolve(A_35[1,:], A_34[:, 2], mode))
            assert npo.allclose(ag_convolve(A_35, A_34, axes=([],[]), dot_axes=([0], [0]), mode=mode),
                                npo.tensordot(A_35, A_34, axes=([0], [0])))
            assert npo.allclose(ag_convolve(A_35, A_342, axes=([1],[2]),
                                            dot_axes=([0], [0]), mode=mode)[2],
                                sum([sp_convolve(A_35[i, :], A_342[i, 2, :], mode)
                                    for i in range(3)]))
            assert npo.allclose(ag_convolve(A_2543, A_24232, axes=([1, 2],[2, 4]),
                                            dot_axes=([0, 3], [0, 3]), mode=mode)[2],
                                sum([sum([sp_convolve(A_2543[i, :, :, j],
                                                    A_24232[i, 2, :, j, :], mode)
                                        for i in range(2)]) for j in range(3)]))

    def test_convolve():
        combo_check(autograd.scipy.signal.convolve, [0,1],
                    [R(4), R(5), R(6)],
                    [R(2), R(3), R(4)], mode=['full', 'valid'])

    def test_convolve_2d():
        combo_check(autograd.scipy.signal.convolve, [0, 1],
                    [R(4, 3), R(5, 4), R(6, 7)],
                    [R(2, 2), R(3, 2), R(4, 2), R(4, 1)], mode=['full', 'valid'])

    def test_convolve_ignore():
        combo_check(autograd.scipy.signal.convolve, [0, 1], [R(4, 3)], [R(3, 2)],
                    axes=[([0],[0]), ([1],[1]), ([0],[1]), ([1],[0]), ([0, 1], [0, 1]), ([1, 0], [1, 0])],
                    mode=['full', 'valid'])

    def test_convolve_ignore_dot():
        combo_check(autograd.scipy.signal.convolve, [0, 1], [R(3, 3, 2)], [R(3, 2, 3)],
                    axes=[([1],[1])], dot_axes=[([0],[2]), ([0],[0])], mode=['full', 'valid'])

    ### Special ###
    def test_polygamma(): combo_check(special.polygamma, [1], [0], R(4)**2 + 1.3)
    def test_jn():        combo_check(special.jn,        [1], [2], R(4)**2 + 1.3)
    def test_yn():        combo_check(special.yn,        [1], [2], R(4)**2 + 1.3)

    def test_psi():       unary_ufunc_check(special.psi,     lims=[0.3, 2.0], test_complex=False)
    def test_digamma():   unary_ufunc_check(special.digamma, lims=[0.3, 2.0], test_complex=False)
    def test_gamma():     unary_ufunc_check(special.gamma,   lims=[0.3, 2.0], test_complex=False)
    def test_gammaln():   unary_ufunc_check(special.gammaln, lims=[0.3, 2.0], test_complex=False)
    def test_gammasgn():  unary_ufunc_check(special.gammasgn,lims=[0.3, 2.0], test_complex=False)
    def test_rgamma()  :  unary_ufunc_check(special.rgamma,  lims=[0.3, 2.0], test_complex=False)
    def test_multigammaln(): combo_check(special.multigammaln, [0], [U(4., 5.), U(4., 5., (2,3))],
                                        [1, 2, 3])

    def test_j0(): unary_ufunc_check(special.j0, lims=[0.2, 20.0], test_complex=False)
    def test_j1(): unary_ufunc_check(special.j1, lims=[0.2, 20.0], test_complex=False)
    def test_y0(): unary_ufunc_check(special.y0, lims=[0.2, 20.0], test_complex=False)
    def test_y1(): unary_ufunc_check(special.y1, lims=[0.2, 20.0], test_complex=False)

    def test_erf(): unary_ufunc_check(special.erf, lims=[-3., 3.], test_complex=True)
    def test_erfc(): unary_ufunc_check(special.erfc, lims=[-3., 3.], test_complex=True)

    def test_erfinv(): unary_ufunc_check(special.erfinv, lims=[-0.95, 0.95], test_complex=False)
    def test_erfcinv(): unary_ufunc_check(special.erfcinv, lims=[0.05, 1.95], test_complex=False)

    def test_logit(): unary_ufunc_check(special.logit, lims=[0.05, 0.95],  test_complex=False)
    def test_expit(): unary_ufunc_check(special.expit, lims=[-4.05, 4.95], test_complex=False)

    ### Linalg ###
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
