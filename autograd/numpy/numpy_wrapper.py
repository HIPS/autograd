from __future__ import absolute_import
import types
import numpy as np
from collections import Iterable
from autograd.container_types import arg_tuple
from autograd.core import primitive, Node

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
            new[name] = primitive(obj)
        elif type(obj) in unchanged_types:
            new[name] = obj

wrap_namespace(np.__dict__, globals())

# ----- Special treatment of list-input functions -----

@primitive
def concatenate_args(axis, *args):
    return np.concatenate(args, axis).view(ndarray)
concatenate = lambda arr_list, axis=0 : concatenate_args(axis, *arr_list)

@primitive
def array_primitive(A, *args, **kwargs):
    return np.array(A, *args, **kwargs).view(ndarray)
array_primitive.defgrad(lambda ans, A : lambda g : g)

def array(A, *args, **kwargs):
    if not isinstance(A, (Node, np.ndarray)) and isinstance(A, Iterable):
        A = recursive_arg_tuple(*A)
    return array_primitive(A, *args, **kwargs)
