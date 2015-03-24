from __future__ import absolute_import
from .numpy_wrapper import wrap_namespace
import numpy.random as npr

wrap_namespace(npr.__dict__, globals())
