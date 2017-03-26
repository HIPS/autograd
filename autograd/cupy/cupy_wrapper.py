from __future__ import absolute_import
import cupy as _cupy
import types
from future.utils import iteritems
from autograd.core import primitive, nograd_primitive, getval
from autograd.numpy.numpy_wrapper import unbox_args, wrap_intdtype

nograd_functions = [
    _cupy.floor, _cupy.ceil, _cupy.rint, _cupy.trunc, _cupy.all, _cupy.any,
    _cupy.argmax, _cupy.argmin, _cupy.nonzero, _cupy.flatnonzero,
    _cupy.count_nonzero, _cupy.sign, _cupy.floor_divide, _cupy.logical_and,
    _cupy.logical_or, _cupy.logical_not, _cupy.logical_xor, _cupy.isfinite,
    _cupy.isinf, _cupy.isnan, _cupy.greater, _cupy.greater_equal, _cupy.less,
    _cupy.less_equal, _cupy.equal, _cupy.not_equal, _cupy.isscalar,
    _cupy.zeros_like, _cupy.ones_like]

def wrap_namespace(old, new):
    unchanged_types = {float, int, type(None), type}
    int_types = {_cupy.int8, _cupy.int16, _cupy.int32, _cupy.int64, _cupy.integer}
    function_types = {_cupy.ufunc, types.FunctionType, types.BuiltinFunctionType}
    for name, obj in iteritems(old):
        if obj in nograd_functions:
            new[name] = nograd_primitive(obj)
        elif type(obj) in function_types:
            new[name] = primitive(obj)
        elif type(obj) is type and obj in int_types:
            new[name] = wrap_intdtype(obj)
        elif type(obj) in unchanged_types:
            new[name] = obj

wrap_namespace(_cupy.__dict__, globals())

# special treatment of list-input functions

@primitive
def concatenate_args(axis, *args):
    return _cupy.concatenate(args, axis)
concatenate = lambda arr_list, axis=0: concatenate_args(axis, *arr_list)

# cupy doesn't actually have an ndim function, but it's convenient
def ndim(x): return x.ndim if not isscalar(x) else 0
