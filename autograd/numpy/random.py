from __future__ import absolute_import
from .numpy_wrapper import wrap_output
import numpy.random as npr
from numpy.random import *
# Objects in numpy.random.__dict__ not imported by *:
mtrand          = npr.mtrand
RandomState     = npr.RandomState
choice          = npr.choice
dirichlet       = npr.dirichlet

W = wrap_output
randn = W(randn)
