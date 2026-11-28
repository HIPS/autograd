import typing as _typing

import numpy.random as npr

from .numpy_wrapper import wrap_namespace

if _typing.TYPE_CHECKING:
    from numpy.random import *

wrap_namespace(npr.__dict__, globals())
