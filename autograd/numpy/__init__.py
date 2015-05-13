from __future__ import absolute_import
from .use_gpu_numpy import use_gpu_numpy
from . import numpy_wrapper
from . import numpy_grads
from . import numpy_extra
from . import complex_array_node

if use_gpu_numpy():
    from . import gpu_array_node

from .numpy_wrapper import *
from . import linalg
from . import fft
from . import random
