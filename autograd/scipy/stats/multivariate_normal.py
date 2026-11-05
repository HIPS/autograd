import scipy.stats

import autograd.numpy as np
from autograd.extend import defvjp, primitive
from autograd.numpy.numpy_vjps import unbroadcast_f

pdf = primitive(scipy.stats.multivariate_normal.pdf)
logpdf = primitive(scipy.stats.multivariate_normal.logpdf)
entropy = primitive(scipy.stats.multivariate_normal.entropy)

# With thanks to Eric Bresch.
# Some formulas are from
# "An extended collection of matrix derivative results
#  for forward and reverse mode algorithmic differentiation"
# by Mike Giles
# https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf


def generalized_outer_product(x):
    if np.ndim(x) == 1:
        return np.outer(x, x)
    return np.matmul(x, np.swapaxes(x, -1, -2))


def covgrad(x, mean, cov, allow_singular=False):
    if allow_singular:
        raise NotImplementedError(
            "The multivariate normal pdf is not differentiable w.r.t. a singular covariance matix"
        )
    J = np.linalg.inv(cov)
    solved = np.matmul(J, np.expand_dims(x - mean, -1))
    return 1.0 / 2 * (generalized_outer_product(solved) - J)


def solve(allow_singular):
    if allow_singular:
        solver = lambda A, x: np.dot(np.linalg.pinv(A), x)
    else:
        solver = np.linalg.solve

    def solve_batch(A, x):
        shape = np.shape(x)
        rhs = np.reshape(x, (-1, shape[-1])).T
        return np.reshape(solver(A, rhs).T, shape)

    return solve_batch


def reshape_weights(g, x, mean):
    return np.reshape(g, np.shape(x - mean)[:-1] + (1,))


defvjp(
    logpdf,
    lambda ans, x, mean, cov, allow_singular=False: unbroadcast_f(
        x, lambda g: -reshape_weights(g, x, mean) * solve(allow_singular)(cov, x - mean)
    ),
    lambda ans, x, mean, cov, allow_singular=False: unbroadcast_f(
        mean, lambda g: reshape_weights(g, x, mean) * solve(allow_singular)(cov, x - mean)
    ),
    lambda ans, x, mean, cov, allow_singular=False: unbroadcast_f(
        cov,
        lambda g: np.expand_dims(reshape_weights(g, x, mean), -1) * covgrad(x, mean, cov, allow_singular),
    ),
)

# Same as log pdf, but multiplied by the pdf (ans).
defvjp(
    pdf,
    lambda ans, x, mean, cov, allow_singular=False: unbroadcast_f(
        x, lambda g: -reshape_weights(ans * g, x, mean) * solve(allow_singular)(cov, x - mean)
    ),
    lambda ans, x, mean, cov, allow_singular=False: unbroadcast_f(
        mean,
        lambda g: reshape_weights(ans * g, x, mean) * solve(allow_singular)(cov, x - mean),
    ),
    lambda ans, x, mean, cov, allow_singular=False: unbroadcast_f(
        cov,
        lambda g: (
            np.expand_dims(reshape_weights(ans * g, x, mean), -1) * covgrad(x, mean, cov, allow_singular)
        ),
    ),
)

defvjp(entropy, None, lambda ans, mean, cov: unbroadcast_f(cov, lambda g: 0.5 * g * np.linalg.inv(cov).T))
