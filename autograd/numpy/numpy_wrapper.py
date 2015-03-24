from __future__ import absolute_import
import numpy as np
from autograd.core import primitive

def keep_keepdims(fun, funname):
    # Numpy doesn't support keepdims for subclasses so this is the workaround
    def new_fun(*args, **kwargs):
        x = args[0]
        return getattr(x, funname)(*args[1:], **kwargs) if isinstance(x, np.ndarray) else x
    return new_fun

def wrap_output(fun):
    # Not all numpy functions preserve the ndarray subclass
    def wrapped_fun(*args, **kwargs):
        ans = fun(*args, **kwargs)
        if isinstance(ans, np.ndarray):
            ans = ans.view(ndarray)
        return ans
    return wrapped_fun

grad_only = ['abs', 'exp', 'log', 'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh',
             'square', 'sqrt', 'sign', 'reshape', 'max', 'dot', 'prod',
             'squeeze', 'repeat', 'transpose', 'split', 'trace']
direct_import = ['float64', 'allclose', 'round', 'argmax', 'unravel_index']
grad_and_wrap = ['outer', 'full', 'ravel', 'expand_dims', 'diag']
grad_and_keepdims = ['sum', 'mean']
wrap_only = ['zeros', 'ones', 'eye']

for x in grad_only:
    globals()[x] = primitive(np.__dict__[x])

for x in direct_import:
    globals()[x] = np.__dict__[x]

for x in grad_and_wrap:
    globals()[x] = primitive(wrap_output(np.__dict__[x]))

for x in grad_and_keepdims:
    globals()[x] = primitive(keep_keepdims(np.__dict__[x], x))

for x in wrap_only:
    globals()[x] = wrap_output(np.__dict__[x])

# ----- Slightly modified version of ndarray -----

class ndarray(np.ndarray):
    def __array_wrap__(self, obj):
        if obj.shape == ():
            return obj[()] # Restoring behavior of regular ndarray
        else:
            return np.ndarray.__array_wrap__(self, obj)

    # Wrap binary ops since the other operand could be a Node
    __add__  = primitive(np.ndarray.__add__)
    __sub__  = primitive(np.ndarray.__sub__)
    __mul__  = primitive(np.ndarray.__mul__)
    __pow__  = primitive(np.ndarray.__pow__)
    __div__  = primitive(np.ndarray.__div__)
    __radd__ = primitive(np.ndarray.__radd__)
    __rsub__ = primitive(np.ndarray.__rsub__)
    __rmul__ = primitive(np.ndarray.__rmul__)
    __rpow__ = primitive(np.ndarray.__rpow__)
    __rdiv__ = primitive(np.ndarray.__rdiv__)

# ----- Special treatment of list-input functions -----

concatenate_args = primitive(wrap_output(lambda axis, *args : np.concatenate(args, axis)))
concatenate = lambda arr_list, axis=0 : concatenate_args(axis, *arr_list)
