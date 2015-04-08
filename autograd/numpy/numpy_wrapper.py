from __future__ import absolute_import
import types
import numpy as np
from autograd.core import primitive

def wrap_namespace(old, new):
    unchanged_types =  set([types.FloatType, types.IntType, types.NoneType, types.TypeType])
    function_types = set([np.ufunc, types.FunctionType, types.BuiltinFunctionType])
    for name, obj in old.iteritems():
        if type(obj) in function_types:
            new[name] = primitive(obj)
        elif type(obj) in unchanged_types:
            new[name] = obj

wrap_namespace(np.__dict__, globals())

# ----- Special treatment of list-input functions -----

@primitive
def concatenate_args(axis, *args):
    return np.concatenate(args, axis).view(ndarray)
concatenate = lambda arr_list, axis=0 : concatenate_args(axis, *arr_list)

def array(A, *args, **kwargs):
    if isinstance(A, np.ndarray):
        return np.array(A, *args, **kwargs)
    else:
        raw_array = np.array(A, *args, **kwargs)
        if raw_array.dtype is np.dtype('O'):
            return array_from_args(raw_array.shape, *raw_array.ravel())
        else:
            return raw_array

@primitive
def array_from_args(front_shape, *args):
    new_array = np.array(args)
    return new_array.reshape(front_shape + new_array.shape[1:])

def array_from_args_gradmaker(argnum, ans, front_shape, *args):
    new_shape = (np.prod(front_shape),) + ans.shape[len(front_shape):]
    return lambda g : reshape(g, new_shape)[argnum - 1]
array_from_args.gradmaker = array_from_args_gradmaker
