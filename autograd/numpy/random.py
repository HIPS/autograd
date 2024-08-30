import numpy.random as npr

from .numpy_wrapper import wrap_namespace

wrap_namespace(npr.__dict__, globals())
