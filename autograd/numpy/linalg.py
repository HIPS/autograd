from __future__ import absolute_import
import numpy.linalg as npla
from .numpy_wrapper import wrap_namespace, dot

wrap_namespace(npla.__dict__, globals())

# Formulas from
# "An extended collection of matrix derivative results
#  for forward and reverse mode algorithmic differentiation"
# by Mike Giles
# https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
inv.defgrad(  lambda ans, x    : lambda g : -dot(dot(ans.T, g), ans.T))
det.defgrad(  lambda ans, x    : lambda g : g * ans * inv(x).T)
solve.defgrad(lambda ans, a, b : lambda g : -dot(solve(a.T, g), ans.T))
solve.defgrad(lambda ans, a, b : lambda g : solve(a.T, g), argnum=1)
norm.defgrad( lambda ans, a    : lambda g : dot(g, a/ans))