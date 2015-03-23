from __future__ import absolute_import
import numpy as np
from autograd.core import primitive as P

# ----- Wrap numpy functions -----

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

W = wrap_output
# Differentiable functions
abs    = P(np.abs)
exp    = P(np.exp)
log    = P(np.log)
sin    = P(np.sin)
cos    = P(np.cos)
tan    = P(np.tan)
sinh   = P(np.sinh)
cosh   = P(np.cosh)
tanh   = P(np.tanh)
square = P(np.square)
sqrt   = P(np.sqrt)
sign   = P(np.sign)
full   = P(W(np.full))
reshape  = P(np.reshape)
ravel    = P(W(np.ravel))
expand_dims = P(W(np.expand_dims))
squeeze     = P(np.squeeze)
repeat      = P(np.repeat)
transpose   = P(np.transpose)
split       = P(np.split)
diag        = P(W(np.diag))
trace       = P(np.trace)
sum         = P(keep_keepdims(np.sum,  'sum'))
max         = P(np.max)
mean        = P(keep_keepdims(np.mean, 'mean'))
dot         = P(np.dot)
prod  = P(np.prod)
outer = P(W(np.outer))

# Functions constant w.r.t. real-valued inputs
float64 = np.float64
allclose = np.allclose
round = np.round
argmax = np.argmax
unravel_index = np.unravel_index
zeros = W(np.zeros)
ones  = W(np.ones)
eye   = W(np.eye)

# ----- Slightly modified version of ndarray -----

class ndarray(np.ndarray):
    def __array_wrap__(self, obj):
        if obj.shape == ():
            return obj[()] # Restoring behavior of regular ndarray
        else:
            return np.ndarray.__array_wrap__(self, obj)

    # Wrap binary ops since the other operand could be a Node
    __add__  = P(np.ndarray.__add__)
    __sub__  = P(np.ndarray.__sub__)
    __mul__  = P(np.ndarray.__mul__)
    __pow__  = P(np.ndarray.__pow__)
    __div__  = P(np.ndarray.__div__)
    __radd__ = P(np.ndarray.__radd__)
    __rsub__ = P(np.ndarray.__rsub__)
    __rmul__ = P(np.ndarray.__rmul__)
    __rpow__ = P(np.ndarray.__rpow__)
    __rdiv__ = P(np.ndarray.__rdiv__)

# ----- Special treatment of list-input functions -----

concatenate_args = P(W(lambda axis, *args : np.concatenate(args, axis)))
concatenate = lambda arr_list, axis=0 : concatenate_args(axis, *arr_list)
