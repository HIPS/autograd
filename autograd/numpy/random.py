from __future__ import absolute_import
from .numpy_core import wrap_output
from numpy.random import *
import numpy.random as npr
# Objects in numpy.random.__dict__ not imported by *:
mtrand          = npr.mtrand
RandomState     = npr.RandomState
choice          = npr.choice
dirichlet       = npr.dirichlet

W = wrap_output
randn = W(randn)
