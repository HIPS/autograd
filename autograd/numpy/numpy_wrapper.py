from __future__ import absolute_import
from __future__ import print_function
import types
from future.utils import iteritems
import warnings
from autograd.core import primitive, nograd_primitive, getval
import numpy as np

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

nograd_functions = [
    np.floor, np.ceil, np.round, np.rint, np.around, np.fix, np.trunc, np.all,
    np.any, np.argmax, np.argmin, np.argpartition, np.argsort, np.argwhere, np.nonzero,
    np.flatnonzero, np.count_nonzero, np.searchsorted, np.sign, np.ndim, np.shape,
    np.floor_divide, np.logical_and, np.logical_or, np.logical_not, np.logical_xor,
    np.isfinite, np.isinf, np.isnan, np.isneginf, np.isposinf, np.allclose, np.isclose,
    np.array_equal, np.array_equiv, np.greater, np.greater_equal, np.less, np.less_equal,
    np.equal, np.not_equal, np.iscomplexobj, np.iscomplex, np.size, np.isscalar,
    np.isreal, np.zeros_like, np.ones_like]

def wrap_namespace(old, new):
    unchanged_types = {float, int, type(None), type}
    int_types = {np.int, np.int8, np.int16, np.int32, np.int64, np.integer}
    function_types = {np.ufunc, types.FunctionType, types.BuiltinFunctionType}
    for name, obj in iteritems(old):
        if obj in nograd_functions:
            new[name] = nograd_primitive(obj)
        elif type(obj) in function_types:
            new[name] = primitive(obj)
        elif type(obj) is type and obj in int_types:
            new[name] = wrap_intdtype(obj)
        elif type(obj) in unchanged_types:
            new[name] = obj

wrap_namespace(np.__dict__, globals())

# ----- Special treatment of list-input functions -----

@primitive
def concatenate_args(axis, *args):
    return np.concatenate(args, axis).view(ndarray)
concatenate = lambda arr_list, axis=0 : concatenate_args(axis, *arr_list)
vstack = lambda tup: concatenate([atleast_2d(_m) for _m in tup], axis=0)
def hstack(tup):
    arrs = [atleast_1d(_m) for _m in tup]
    if arrs[0].ndim == 1:
        return concatenate(arrs, 0)
    return concatenate(arrs, 1)

def array(A, *args, **kwargs):
    if isinstance(A, np.ndarray):
        return np.array(A, *args, **kwargs)
    else:
        raw_array = np.array(A, *args, **kwargs)
        return wrap_if_nodes_inside(raw_array)

def wrap_if_nodes_inside(raw_array, slow_op_name=None):
    if raw_array.dtype is np.dtype('O'):
        if slow_op_name:
            warnings.warn("{0} is slow for array inputs. "
                          "np.concatenate() is faster.".format(slow_op_name))
        return array_from_args(*raw_array.ravel()).reshape(raw_array.shape)
    else:
        return raw_array

@primitive
def array_from_args(*args):
    return np.array(args)

def array_from_args_gradmaker(argnum, g, ans, vs, gvs, args, kwargs):
    return g[argnum]
array_from_args.grad = array_from_args_gradmaker

def select(condlist, choicelist, default=0):
    raw_array = np.select(list(condlist), list(choicelist), default=default)
    return array(list(raw_array.ravel())).reshape(raw_array.shape)

# ----- Enable functions called using [] ----

class r_class():
    def __getitem__(self, args):
        raw_array = np.r_[args]
        return wrap_if_nodes_inside(raw_array, slow_op_name = "r_")
r_ = r_class()

class c_class():
    def __getitem__(self, args):
        raw_array = np.c_[args]
        return wrap_if_nodes_inside(raw_array, slow_op_name = "c_")
c_ = c_class()
