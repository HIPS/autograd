from __future__ import absolute_import
import types
import numpy as np
import inspect
from autograd.core import primitive, differentiable_ops, nondifferentiable_ops

def keep_keepdims(fun, funname):
    # Numpy doesn't support keepdims for subclasses so this is the workaround
    def new_fun(*args, **kwargs):
        x = args[0]
        if isinstance(x, np.ndarray):
            return getattr(x, funname)(*args[1:], **kwargs)
        else:
            return fun(*args, **kwargs)
    new_fun.__name__ = fun.__name__
    return new_fun
keepdims_stats_funs = ['all', 'any', 'max', 'mean', 'min', 'prod', 'std', 'sum', 'var']

def numpy_wrap(fun):
    # Not all numpy functions preserve the ndarray subclass
    def wrapped_fun(*args, **kwargs):
        ans = fun(*args, **kwargs)
        if type(ans) is np.ndarray:
            ans = ans.view(ndarray)
        return ans
    wrapped_fun.__name__ = fun.__name__
    return wrapped_fun

def wrap_namespace(old, new):
    unchanged_types =  set([types.FloatType, types.IntType, types.NoneType, types.TypeType])
    function_types = set([np.ufunc, types.FunctionType, types.BuiltinFunctionType])
    for name, obj in old.iteritems():
        if type(obj) in function_types:
            if name in keepdims_stats_funs:
                obj = keep_keepdims(obj, name)
            new[name] = primitive(numpy_wrap(obj))
        elif type(obj) in unchanged_types:
            new[name] = obj

wrap_namespace(np.__dict__, globals())

# ----- Slightly modified version of ndarray -----

class ndarray(np.ndarray):
    def __array_wrap__(self, obj):
        if obj.shape == ():
            return obj[()] # Restoring behavior of regular ndarray
        else:
            return np.ndarray.__array_wrap__(self, obj)
    dot = dot

# Wrap binary ops since the other operand could be a Node
for ndarray_op in differentiable_ops + nondifferentiable_ops:
    setattr(ndarray, ndarray_op, primitive(getattr(ndarray, ndarray_op)))

# ----- Special treatment of list-input functions -----

concatenate_args = primitive(numpy_wrap(lambda axis, *args : np.concatenate(args, axis)))
concatenate = lambda arr_list, axis=0 : concatenate_args(axis, *arr_list)
