from __future__ import absolute_import
import scipy.stats

import autograd.numpy as np
from autograd.numpy.util import unbroadcast_f
from autograd.extend import primitive, defvjp


pdf    =  primitive(scipy.stats.multivariate_normal.pdf)
logpdf =  primitive(scipy.stats.multivariate_normal.logpdf)
entropy = primitive(scipy.stats.multivariate_normal.entropy)

# With thanks to Eric Bresch.
# Some formulas are from
# "An extended collection of matrix derivative results
#  for forward and reverse mode algorithmic differentiation"
# by Mike Giles
# https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf

def lower_half(mat):
    # Takes the lower half of the matrix, and half the diagonal.
    # Necessary since numpy only uses lower half of covariance matrix.
    if len(mat.shape) == 2:
        return 0.5 * (np.tril(mat) + np.triu(mat, 1).T)
    elif len(mat.shape) == 3:
        return 0.5 * (np.tril(mat) + np.swapaxes(np.triu(mat, 1), 1,2))
    else:
        raise ArithmeticError

def generalized_outer_product(mat):
    if len(mat.shape) == 1:
        return np.outer(mat, mat)
    elif len(mat.shape) == 2:
        return np.einsum('ij,ik->ijk', mat, mat)
    else:
        raise ArithmeticError

def covgrad(x, mean, cov, allow_singular=False):
    if allow_singular:
        raise NotImplementedError("The multivariate normal pdf is not "
                "differentiable w.r.t. a singular covariance matix")
    # I think once we have Cholesky we can make this nicer.
    solved = np.linalg.solve(cov, (x - mean).T).T
    return lower_half(np.linalg.inv(cov) - generalized_outer_product(solved))

def solve(allow_singular):
    if allow_singular:
        return lambda A, x: np.dot(np.linalg.pinv(A), x)
    else:
        return np.linalg.solve

defvjp(logpdf,
       lambda ans, x, mean, cov, allow_singular=False:
       unbroadcast_f(x, lambda g: -np.expand_dims(g, 1) * solve(allow_singular)(cov, (x - mean).T).T),
       lambda ans, x, mean, cov, allow_singular=False:
       unbroadcast_f(mean, lambda g:  np.expand_dims(g, 1) * solve(allow_singular)(cov, (x - mean).T).T),
       lambda ans, x, mean, cov, allow_singular=False:
       unbroadcast_f(cov, lambda g: -np.reshape(g, np.shape(g) + (1, 1)) * covgrad(x, mean, cov, allow_singular)))

# Same as log pdf, but multiplied by the pdf (ans).
defvjp(pdf,
       lambda ans, x, mean, cov, allow_singular=False:
       unbroadcast_f(x, lambda g: -np.expand_dims(ans * g, 1) * solve(allow_singular)(cov, (x - mean).T).T),
       lambda ans, x, mean, cov, allow_singular=False:
       unbroadcast_f(mean, lambda g:  np.expand_dims(ans * g, 1) * solve(allow_singular)(cov, (x - mean).T).T),
       lambda ans, x, mean, cov, allow_singular=False:
       unbroadcast_f(cov, lambda g: -np.reshape(ans * g, np.shape(g) + (1, 1)) * covgrad(x, mean, cov, allow_singular)))

defvjp(entropy, None,
       lambda ans, mean, cov:
       unbroadcast_f(cov, lambda g: 0.5 * g * np.linalg.inv(cov).T))
