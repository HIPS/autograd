from __future__ import absolute_import
from __future__ import print_function
import types
from .use_gpu_numpy import use_gpu_numpy
from future.utils import iteritems


if use_gpu_numpy():
    print("Using GPU-supporting numpy wrapper")
    import gpu_numpy as _np
else:
    import numpy as _np

import warnings
from autograd.core import primitive, getval

def unbox_args(f):
    def wrapped(*args, **kwargs):
        unboxed_args = map(getval, args)
        unboxed_kwargs = {key: getval(kwargs[key]) for key in kwargs}
        return f(*unboxed_args, **unboxed_kwargs)
    return wrapped

def wrap_intdtype(cls):
    class IntdtypeSubclass(cls):
        __new__ = unbox_args(cls.__new__)
    return IntdtypeSubclass

def wrap_namespace(old, new):
    unchanged_types = {float, int, type(None), type}
    int_types = {_np.int, _np.int8, _np.int16, _np.int32, _np.int64, _np.integer}
    function_types = {_np.ufunc, types.FunctionType, types.BuiltinFunctionType}
    for name, obj in iteritems(old):
        if type(obj) in function_types:
            new[name] = primitive(obj)
        elif type(obj) is type and obj in int_types:
            new[name] = wrap_intdtype(obj)
        elif type(obj) in unchanged_types:
            new[name] = obj

wrap_namespace(_np.__dict__, globals())

# ----- Special treatment of list-input functions -----

@primitive
def concatenate_args(axis, *args):
    return _np.concatenate(args, axis).view(ndarray)
concatenate = lambda arr_list, axis=0 : concatenate_args(axis, *arr_list)
vstack = row_stack = lambda tup: concatenate([atleast_2d(_m) for _m in tup], axis=0)
def hstack(tup):
    arrs = [atleast_1d(_m) for _m in tup]
    if arrs[0].ndim == 1:
        return concatenate(arrs, 0)
    return concatenate(arrs, 1)

def column_stack(tup):
    arrays = []
    for v in tup:
        arr = array(v)
        if arr.ndim < 2:
            arr = array(arr, ndmin=2).T
        arrays.append(arr)
    return concatenate(arrays, 1)

def array(A, *args, **kwargs):
    if isinstance(A, _np.ndarray):
        return _np.array(A, *args, **kwargs)
    else:
        raw_array = _np.array(A, *args, **kwargs)
        return wrap_if_nodes_inside(raw_array)

def wrap_if_nodes_inside(raw_array, slow_op_name=None):
    if raw_array.dtype is _np.dtype('O'):
        if slow_op_name:
            warnings.warn("{0} is slow for array inputs. "
                          "np.concatenate() is faster.".format(slow_op_name))
        return array_from_args(*raw_array.ravel()).reshape(raw_array.shape)
    else:
        return raw_array

@primitive
def array_from_args(*args):
    return _np.array(args)

def array_from_args_gradmaker(argnum, ans, args, kwargs):
    return lambda g : g[argnum]
array_from_args.gradmaker = array_from_args_gradmaker

def select(condlist, choicelist, default=0):
    raw_array = _np.select(list(condlist), list(choicelist), default=default)
    return array(list(raw_array.ravel())).reshape(raw_array.shape)

def stack(arrays, axis=0):
    # this code is basically copied from numpy/core/shape_base.py's stack

    arrays = [array(arr) for arr in arrays]
    if not arrays:
        raise ValueError('need at least one array to stack')

    shapes = set(arr.shape for arr in arrays)
    if len(shapes) != 1:
        raise ValueError('all input arrays must have the same shape')

    result_ndim = arrays[0].ndim + 1
    if not -result_ndim <= axis < result_ndim:
        raise IndexError('axis {0} out of bounds [-{1}, {1})'.format(axis, result_ndim))
    if axis < 0:
        axis += result_ndim

    sl = (slice(None),) * axis + (None,)
    return concatenate([arr[sl] for arr in arrays], axis=axis)

# ----- Enable functions called using [] ----

class r_class():
    def __getitem__(self, args):
        raw_array = _np.r_[args]
        return wrap_if_nodes_inside(raw_array, slow_op_name = "r_")
r_ = r_class()

class c_class():
    def __getitem__(self, args):
        raw_array = _np.c_[args]
        return wrap_if_nodes_inside(raw_array, slow_op_name = "c_")
c_ = c_class()
