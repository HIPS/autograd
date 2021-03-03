from __future__ import absolute_import
import types
import warnings
from autograd.extend import primitive, notrace_primitive
import numpy as _np
import autograd.builtins as builtins
from numpy.core.einsumfunc import _parse_einsum_input

notrace_functions = [
    _np.ndim, _np.shape, _np.iscomplexobj, _np.result_type
]

def wrap_intdtype(cls):
    class IntdtypeSubclass(cls):
        __new__ = notrace_primitive(cls.__new__)
    return IntdtypeSubclass

def wrap_namespace(old, new):
    unchanged_types = {float, int, type(None), type}
    int_types = {_np.int8, _np.int16, _np.int32, _np.int64, _np.integer}
    for name, obj in old.items():
        if obj in notrace_functions:
            new[name] = notrace_primitive(obj)
        elif callable(obj) and type(obj) is not type:
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
    t = builtins.type(A)
    if t in (list, tuple):
        return array_from_args(args, kwargs, *map(array, A))
    else:
        return _array_from_scalar_or_array(args, kwargs, A)

def wrap_if_boxes_inside(raw_array, slow_op_name=None):
    if raw_array.dtype is _np.dtype('O'):
        if slow_op_name:
            warnings.warn("{0} is slow for array inputs. "
                          "np.concatenate() is faster.".format(slow_op_name))
        return array_from_args((), {}, *raw_array.ravel()).reshape(raw_array.shape)
    else:
        return raw_array

@primitive
def _array_from_scalar_or_array(array_args, array_kwargs, scalar):
    return _np.array(scalar, *array_args, **array_kwargs)

@primitive
def array_from_args(array_args, array_kwargs, *args):
    return _np.array(args, *array_args, **array_kwargs)

def select(condlist, choicelist, default=0):
    raw_array = _np.select(list(condlist), list(choicelist), default=default)
    return array(list(raw_array.ravel())).reshape(raw_array.shape)

def stack(arrays, axis=0):
    # this code is basically copied from numpy/core/shape_base.py's stack
    # we need it here because we want to re-implement stack in terms of the
    # primitives defined in this file

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

def append(arr, values, axis=None):
    # this code is basically copied from numpy/lib/function_base.py's append
    arr = array(arr)
    if axis is None:
        if ndim(arr) != 1:
            arr = ravel(arr)
        values = ravel(array(values))
        axis = ndim(arr) - 1
    return concatenate((arr, values), axis=axis)

# ----- Enable functions called using [] ----

class r_class():
    def __getitem__(self, args):
        raw_array = _np.r_[args]
        return wrap_if_boxes_inside(raw_array, slow_op_name = "r_")
r_ = r_class()

class c_class():
    def __getitem__(self, args):
        raw_array = _np.c_[args]
        return wrap_if_boxes_inside(raw_array, slow_op_name = "c_")
c_ = c_class()

# ----- misc -----
@primitive
def make_diagonal(D, offset=0, axis1=0, axis2=1):
    # Numpy doesn't offer a complement to np.diagonal: a function to create new
    # diagonal arrays with extra dimensions. We need such a function for the
    # gradient of np.diagonal and it's also quite handy to have. So here it is.
    if not (offset==0 and axis1==-1 and axis2==-2):
        raise NotImplementedError("Currently make_diagonal only supports offset=0, axis1=-1, axis2=-2")

    # We use a trick: calling np.diagonal returns a view on the original array,
    # so we can modify it in-place. (only valid for numpy version >= 1.10.)
    new_array = _np.zeros(D.shape + (D.shape[-1],))
    new_array_diag = _np.diagonal(new_array, offset=0, axis1=-1, axis2=-2)
    new_array_diag.flags.writeable = True
    new_array_diag[:] = D
    return new_array

@notrace_primitive
def metadata(A):
    return _np.shape(A), _np.ndim(A), _np.result_type(A), _np.iscomplexobj(A)

@notrace_primitive
def parse_einsum_input(*args):
    return _parse_einsum_input(args)

@primitive
def _astype(A, dtype, order='K', casting='unsafe', subok=True, copy=True):
  return A.astype(dtype, order, casting, subok, copy)
