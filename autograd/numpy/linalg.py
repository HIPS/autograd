from __future__ import absolute_import
from numpy.linalg import *
from autograd.core import primitive
from .numpy_core import dot

# ----- Gradients -----
P = primitive
inv  = P(inv,  lambda ans, x : [lambda g : -dot(dot(ans.T, g), ans.T)])
det  = P(det,  lambda ans, x : [lambda g : g * ans * inv(x).T])
