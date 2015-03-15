from .numpy_core import *
# Without these del statements the wrapper imports will fail
del(linalg)
del(random)
from . import linalg
from . import random
