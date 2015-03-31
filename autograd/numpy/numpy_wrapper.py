from __future__ import absolute_import
import types
import numpy as np
import inspect
from collections import Iterable
from autograd.container_types import arg_tuple
from autograd.core import primitive, Node, differentiable_ops, nondifferentiable_ops

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
        return safe_type(fun(*args, **kwargs))
    wrapped_fun.__name__ = fun.__name__
    return wrapped_fun

def safe_type(x):
    if type(x) is ndarray and x.shape == ():
        return x[()] # Restoring behavior of regular ndarray
    elif type(x) is np.ndarray:
        return x.view(ndarray)
    elif type(x) is tuple:
        return tuple([safe_type(a) for a in x])
    else:
        return x

def recursive_arg_tuple(*args):
    args = list(args)
    for i, arg in enumerate(args):
        if isinstance(arg, Iterable):
            args[i] = recursive_arg_tuple(*arg)
    return arg_tuple(*args)

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
    dot = dot
    def __neg__(self): return negative(self)
    def __add__(self, other): return add(     self, other)
    def __sub__(self, other): return subtract(self, other)
    def __mul__(self, other): return multiply(self, other)
    def __pow__(self, other): return power   (self, other)
    def __div__(self, other): return divide(  self, other)
    def __radd__(self, other): return add(     other, self)
    def __rsub__(self, other): return subtract(other, self)
    def __rmul__(self, other): return multiply(other, self)
    def __rpow__(self, other): return power   (other, self)
    def __rdiv__(self, other): return divide(  other, self)
    def __eq__(self, other): return equal(self, other)
    def __ne__(self, other): return not_equal(self, other)
    def __gt__(self, other): return greater(self, other)
    def __ge__(self, other): return greater_equal(self, other)
    def __lt__(self, other): return less(self, other)
    def __le__(self, other): return less_equal(self, other)

# ----- Special treatment of list-input functions -----

@primitive
def concatenate_args(axis, *args):
    return np.concatenate(args, axis).view(ndarray)
concatenate = lambda arr_list, axis=0 : concatenate_args(axis, *arr_list)

@primitive
def array_primitive(A):
    return np.array(A).view(ndarray)
array_primitive.defgrad(lambda ans, A : lambda g : g)

def array(A):
    if not isinstance(A, (Node, np.ndarray)) and isinstance(A, Iterable):
        A = recursive_arg_tuple(*A)
    return array_primitive(A)
