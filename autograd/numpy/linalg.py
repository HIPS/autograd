from __future__ import absolute_import
from numpy.linalg import *
from autograd.core import primitive
from .numpy_core import dot

# ----- Gradients -----

P = primitive
inv  = P(inv)
det  = P(det)

inv.defgrad(lambda ans, x : lambda g : -dot(dot(ans.T, g), ans.T))
det.defgrad(lambda ans, x : lambda g : g * ans * inv(x).T)
