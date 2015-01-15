import numpy as np
import operator as op
import itertools as it
from functools import partial
from core import Differentiable, getval, untake
D = Differentiable

# ----- Operator gradients -----

op.neg = D(op.neg, [lambda g, x : - g])
op.add = D(op.add, [lambda g, x, y : g,     lambda g, x, y : g])
op.mul = D(op.mul, [lambda g, x, y : y * g, lambda g, x, y : x * g])
op.sub = D(op.sub, [lambda g, x, y : g,     lambda g, x, y : - g])
op.div = D(op.div, [lambda g, x, y : g / y, lambda g, x, y : - g * x / y**2])
op.pow = D(op.pow, [lambda g, x, y : g * y * x ** (y - 1),
                    lambda g, x, y : g * np.log(x) * x ** y])

isarray = lambda x : isinstance(getval(x), np.ndarray)
isfloat = lambda x : isinstance(getval(x), float)

def undo_broadcast(fun, argnum):
    def new_fun(g, *args):
        ans = fun(g, *args)
        x = args[argnum]
        if isfloat(x) and isarray(ans):
            ans = np.sum(ans)
        elif isarray(x):
            while ans.ndim > x.ndim:
                ans = np.sum(ans, axis=0)
            for axis, size in enumerate(x.shape):
                if size is 1:
                    ans = np.sum(ans, axis, keepdims=True)
        return ans

    return new_fun

broadcasting_ops = [op.add, op.mul, op.sub, op.div, op.pow]
for fun, argnum in it.product(broadcasting_ops, [0, 1]):
    fun.gradfuns[argnum] = undo_broadcast(fun.gradfuns[argnum], argnum)

# ----- Numpy gradients -----

np.abs    = D(np.abs,    [lambda g, x : np.sign(x) * g])
np.exp    = D(np.exp,    [lambda g, x : np.exp(x) * g])
np.log    = D(np.log,    [lambda g, x : g / x])
np.sin    = D(np.sin,    [lambda g, x : g * np.cos(x)])
np.cos    = D(np.cos,    [lambda g, x : - g * np.sin(x)])
np.tan    = D(np.tan,    [lambda g, x : g / np.cos(x) **2])
np.sinh   = D(np.sinh,   [lambda g, x : g * np.cosh(x)])
np.cosh   = D(np.cosh,   [lambda g, x : g * np.sinh(x)])
np.tanh   = D(np.tanh,   [lambda g, x : g / np.cosh(x) **2])
np.square = D(np.square, [lambda g, x : g * 2 * x])
np.sign   = D(np.sign,   [lambda g, x : 0.0])
np.full   = D(np.full,   [None, lambda g, shape, fill_value :  np.sum(g)])
np.reshape     = D(np.reshape,     [lambda g, x, shape, order=None: np.reshape(g, x.shape, order=order)])
np.expand_dims = D(np.expand_dims, [lambda g, x, axis : np.squeeze(g, axis)])
np.squeeze     = D(np.squeeze,     [lambda g, x, axis : np.repeat(g, x.shape[axis], axis)])
np.repeat      = D(np.repeat,      [lambda g, x, shape, axis : np.sum(g, axis, keepdims=True)])
np.transpose   = D(np.transpose,   [lambda g, x : np.transpose(g)])
np.split       = D(np.split, [lambda g, A, idxs, axis=0 : np.concatenate(g, axis=axis)])

def grad_np_sum(g, x, axis=None, keepdims=False):
    if not isarray(x):
        return g
    if axis is None:
        return np.full(x.shape, g)
    elif not keepdims:
        g = np.expand_dims(g, axis)
    return np.repeat(g, x.shape[axis], axis)
np.sum = D(np.sum, [grad_np_sum])

def grad_np_max(g, x):
    idxs = np.argmax(getval(x))
    return untake(g, np.unravel_index(idxs, x.shape))
np.max = D(np.max, [grad_np_max])

def grad_np_dot_A(g, A, B):
    if B.ndim is 2:
        return np.dot(g, B.T)
    elif A.ndim is 2:
        return np.outer(g, B)
    else:
        return g * B
def grad_np_dot_B(g, A, B):
    if A.ndim is 2:
        return np.dot(A.T, g)
    elif B.ndim is 2:
        return np.outer(A, g)
    else:
        return g * A
np.dot = D(np.dot, [grad_np_dot_A, grad_np_dot_B])

def grad_np_concatenate(g, arr_list, axis=0):
    idxs = np.cumsum([a.shape[axis] for a in getval(arr_list)[:-1]])
    return np.split(g, idxs, axis=axis)
np.concatenate = D(np.concatenate, [grad_np_concatenate])

# ----- Special list constructor -----

class ArgnumGrad(object):
    def __init__(self, fun_with_argnum):
        self.fun = fun_with_argnum
    def __getitem__(self, argnum):
        return partial(self.fun, argnum)

@Differentiable
def kylist(*args):
    return list(args)
kylist.gradfuns = ArgnumGrad(lambda argnum, g, *args : g[argnum])

# Wrap the concatenation function to automatically wrap the list into a kylist.
unwrapped_np_concatenate = np.concatenate
def concatwrapper(*args, **kwargs):
    args = (kylist(*(args[0])),) + args[1:]
    return unwrapped_np_concatenate(*args, **kwargs)
np.concatenate = concatwrapper
