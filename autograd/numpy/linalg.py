from __future__ import absolute_import
import numpy.linalg as npla
from autograd.core import primitive as P
from .numpy_wrapper import dot

# ----- Gradients -----

inv  = P(npla.inv)
det  = P(npla.det)

inv.defgrad(lambda ans, x : lambda g : -dot(dot(ans.T, g), ans.T))
det.defgrad(lambda ans, x : lambda g : g * ans * inv(x).T)
