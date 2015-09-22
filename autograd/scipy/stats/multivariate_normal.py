from __future__ import absolute_import
import scipy.stats

import autograd.numpy as np
from autograd.core import primitive

pdf    = primitive(scipy.stats.multivariate_normal.pdf)
logpdf = primitive(scipy.stats.multivariate_normal.logpdf)

# With thanks to Eric Bresch.
# Some formulas are from
# "An extended collection of matrix derivative results
#  for forward and reverse mode algorithmic differentiation"
# by Mike Giles
# https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf

def lower_half(mat):
    # Takes the lower half of the matrix, and half the diagonal.
    # Necessary since numpy only uses lower half of covariance matrix.
    return 0.5 * (np.tril(mat) + np.triu(mat, 1).T)

def covgrad(x, mean, cov):
    # I think once we have Cholesky we can make this nicer.
    solved = np.linalg.solve(cov, x - mean)
    return lower_half(np.linalg.inv(cov) - np.outer(solved, solved))

logpdf.defgrad(lambda ans, x, mean, cov: lambda g: -g * np.linalg.solve(cov, x - mean), argnum=0)
logpdf.defgrad(lambda ans, x, mean, cov: lambda g:  g * np.linalg.solve(cov, x - mean), argnum=1)
logpdf.defgrad(lambda ans, x, mean, cov: lambda g: -g * covgrad(x, mean, cov),          argnum=2)

# Same as log pdf, but multiplied by the pdf (ans).
pdf.defgrad(lambda ans, x, mean, cov: lambda g: -g * ans * np.linalg.solve(cov, x - mean), argnum=0)
pdf.defgrad(lambda ans, x, mean, cov: lambda g:  g * ans * np.linalg.solve(cov, x - mean), argnum=1)
pdf.defgrad(lambda ans, x, mean, cov: lambda g: -g * ans * covgrad(x, mean, cov),          argnum=2)
