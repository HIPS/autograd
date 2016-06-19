from __future__ import absolute_import
import scipy.stats

import autograd.numpy as np
from autograd.core import primitive
from autograd.numpy.numpy_grads import unbroadcast

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

def covgrad(x, mean, cov):
    # I think once we have Cholesky we can make this nicer.
    solved = np.linalg.solve(cov, (x - mean).T).T
    return lower_half(np.linalg.inv(cov) - generalized_outer_product(solved))

logpdf.defgrad(lambda ans, x, mean=None, cov=1, allow_singular=False: unbroadcast(ans, x,    lambda g: -np.expand_dims(g, 1) * np.linalg.solve(cov, (x - mean).T).T), argnum=0)
logpdf.defgrad(lambda ans, x, mean=None, cov=1, allow_singular=False: unbroadcast(ans, mean, lambda g:  np.expand_dims(g, 1) * np.linalg.solve(cov, (x - mean).T).T), argnum=1)
logpdf.defgrad(lambda ans, x, mean=None, cov=1, allow_singular=False: unbroadcast(ans, cov,  lambda g: -np.reshape(g, np.shape(g) + (1, 1)) * covgrad(x, mean, cov)), argnum=2)

# Same as log pdf, but multiplied by the pdf (ans).
pdf.defgrad(lambda ans, x, mean=None, cov=1, allow_singular=False: unbroadcast(ans, x,    lambda g: -g * ans * np.linalg.solve(cov, x - mean)), argnum=0)
pdf.defgrad(lambda ans, x, mean=None, cov=1, allow_singular=False: unbroadcast(ans, mean, lambda g:  g * ans * np.linalg.solve(cov, x - mean)), argnum=1)
pdf.defgrad(lambda ans, x, mean=None, cov=1, allow_singular=False: unbroadcast(ans, cov,  lambda g: -g * ans * covgrad(x, mean, cov)),          argnum=2)

entropy.defgrad_is_zero(argnums=(0,))
entropy.defgrad(lambda ans, mean, cov: unbroadcast(ans, cov, lambda g:  0.5 * g * np.linalg.inv(cov).T), argnum=1)
