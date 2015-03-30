from __future__ import absolute_import
import numpy.random as npr
from .numpy_wrapper import wrap_namespace, numpy_wrap

wrap_namespace(npr.__dict__, globals())

class RandomState(npr.RandomState):
    pass

for method_name, method in npr.RandomState.__dict__.iteritems():
    if type(method) == type(npr.RandomState.__dict__['rand']):
        wrapped_method = numpy_wrap(getattr(RandomState, method_name))
        setattr(RandomState, method_name, wrapped_method)
