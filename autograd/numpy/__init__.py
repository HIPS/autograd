# Overwrite where to preserve native Python types
import numpy as _np

from autograd.tracer import Box

from . import fft, linalg, numpy_boxes, numpy_jvps, numpy_vjps, numpy_vspaces, numpy_wrapper, random
from .numpy_wrapper import *
from .numpy_wrapper import numpy_version as __version__

_original_where = getattr(numpy_wrapper, "where")


def _is_np_or_autograd(x):
    if isinstance(x, Box) or isinstance(x, (_np.ndarray, _np.generic)):
        return True
    if isinstance(x, (list, tuple)):
        for item in x:
            if _is_np_or_autograd(item):
                return True
    return False


def custom_where(condition, *args):
    if not args:
        return _original_where(condition)
    if len(args) != 2:
        raise ValueError("either both or neither of x and y should be given")
    x, y = args
    if _is_np_or_autograd(condition) or _is_np_or_autograd(x) or _is_np_or_autograd(y):
        return _original_where(condition, x, y)
    res = _original_where(condition, x, y)
    if isinstance(res, _np.ndarray):
        return res.tolist()
    return res


where = custom_where
setattr(numpy_wrapper, "where", where)
