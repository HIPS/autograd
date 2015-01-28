import numpy as np
import operator as op
import itertools as it
from functools import partial
from core import primitive, getval, untake

P = primitive

# ----- Operator gradients -----
I = lambda x : x # Identity operator
op.neg = P(op.neg, lambda ans, x    : [op.neg])
op.add = P(op.add, lambda ans, x, y : unbroadcast(ans, x, y, [I, I]))
op.mul = P(op.mul, lambda ans, x, y : unbroadcast(ans, x, y, [lambda g : y * g, lambda g : x * g]))
op.sub = P(op.sub, lambda ans, x, y : unbroadcast(ans, x, y, [I, op.neg]))
op.div = P(op.div, lambda ans, x, y : unbroadcast(ans, x, y, [lambda g : g / y, lambda g : - g * x / y**2]))
op.pow = P(op.pow, lambda ans, x, y : unbroadcast(ans, x, y, [lambda g : g * y * x ** (y - 1),
                                                              lambda g : g * np.log(x) * x ** y]))
isarray = lambda x : isinstance(getval(x), np.ndarray)
isfloat = lambda x : isinstance(getval(x), float)

def unbroadcast(ans, x, y, funs):
    return [unbroadcast_fun(ans, x, funs[0]),
            unbroadcast_fun(ans, y, funs[1])]

def unbroadcast_fun(ans, x, fun):
    if isfloat(x) and isarray(ans):
        return lambda g : np.sum(fun(g))
    elif isarray(x):
        shape = x.shape
        def new_fun(g):
            result = fun(g)
            while result.ndim > len(shape):
                result = np.sum(result, axis=0)
            for axis, size in enumerate(shape):
                if size is 1:
                    result = np.sum(result, axis, keepdims=True)
            return result
        return new_fun
    else:
        return fun

# ----- Numpy gradients -----

np.abs    = P(np.abs,    lambda ans, x : [lambda g : np.sign(x) * g])
np.exp    = P(np.exp,    lambda ans, x : [lambda g : ans * g])
np.log    = P(np.log,    lambda ans, x : [lambda g : g / x])
np.sin    = P(np.sin,    lambda ans, x : [lambda g : g * np.cos(x)])
np.cos    = P(np.cos,    lambda ans, x : [lambda g : - g * np.sin(x)])
np.tan    = P(np.tan,    lambda ans, x : [lambda g : g / np.cos(x) **2])
np.sinh   = P(np.sinh,   lambda ans, x : [lambda g : g * np.cosh(x)])
np.cosh   = P(np.cosh,   lambda ans, x : [lambda g : g * np.sinh(x)])
np.tanh   = P(np.tanh,   lambda ans, x : [lambda g : g / np.cosh(x) **2])
np.square = P(np.square, lambda ans, x : [lambda g : g * 2 * x])
np.sign   = P(np.sign,   lambda ans, x : [lambda g : 0.0])
np.full   = P(np.full,   lambda ans, shape, fill_value : [None, lambda g :  np.sum(g)])
np.reshape     = P(np.reshape,     lambda ans, x, shape, order=None : [lambda g : np.reshape(g, x.shape, order=order)])
np.ravel       = P(np.ravel,       lambda ans, x,        order=None : [lambda g : np.reshape(g, x.shape, order=order)])
np.expand_dims = P(np.expand_dims, lambda ans, x, axis              : [lambda g : np.squeeze(g, axis)])
np.squeeze     = P(np.squeeze,     lambda ans, x, axis              : [lambda g : np.repeat(g, x.shape[axis], axis)])
np.repeat      = P(np.repeat,      lambda ans, x, shape, axis       : [lambda g : np.sum(g, axis, keepdims=True)])
np.transpose   = P(np.transpose,   lambda ans, x                    : [lambda g : np.transpose(g)])
np.split       = P(np.split,       lambda ans, A, idxs, axis=0      : [lambda g : np.concatenate(g, axis=axis)])

def make_grad_np_sum(ans, x, axis=None, keepdims=False):
    if not isarray(x):
        return [I]
    shape = x.shape
    if axis is None:
        return [lambda g : np.full(shape, g)]
    else:
        if keepdims:
            return [lambda g : np.repeat(g, shape[axis], axis)]
        else:
            return [lambda g : np.repeat(np.expand_dims(g, axis),
                                         shape[axis], axis)]
np.sum = P(np.sum, make_grad_np_sum)

def make_grad_np_mean(ans, x, axis=None, keepdims=False):
    if not isarray(x):
        return [I]
    shape = x.shape
    if axis is None:
        return [lambda g : np.full(shape, g) / np.prod(shape)]
    else:
        if keepdims:
            return [lambda g : np.repeat(g, shape[axis], axis) / shape[axis]]
        else:
            return [lambda g : np.repeat(np.expand_dims(g, axis),
                                         shape[axis], axis) / shape[axis]]
np.mean = P(np.mean, make_grad_np_mean)

def make_grad_np_max(ans, x):
    def gradfun(g):
        idxs = np.argmax(getval(x))
        return untake(g, np.unravel_index(idxs, x.shape))
    return [gradfun]
np.max = P(np.max, make_grad_np_max)

def make_grad_np_dot(ans, A, B):
    def grad_np_dot_A(g):
        if B.ndim is 2:
            return np.dot(g, B.T)
        elif A.ndim is 2:
            return np.outer(g, B)
        else:
            return g * B
    def grad_np_dot_B(g):
        if A.ndim is 2:
            return np.dot(A.T, g)
        elif B.ndim is 2:
            return np.outer(A, g)
        else:
            return g * A
    return [grad_np_dot_A, grad_np_dot_B]
np.dot = P(np.dot, make_grad_np_dot)

def make_grad_np_concatenate(ans, arr_list, axis=0):
    def grad_np_concatenate(g):
        idxs = np.cumsum([a.shape[axis] for a in getval(arr_list)[:-1]])
        return np.split(g, idxs, axis=axis)
    return [grad_np_concatenate]
np.concatenate = P(np.concatenate, make_grad_np_concatenate)

# ----- Special list constructor -----

class ArgnumGrad(object):
    def __init__(self, fun_with_argnum):
        self.fun = fun_with_argnum
    def __getitem__(self, argnum):
        return partial(self.fun, argnum)

def kylist(*args):
    return list(args)
kylist = primitive(kylist, lambda ans, *args : ArgnumGrad(lambda argnum, g : g[argnum]))

# Wrap the concatenation function to automatically wrap the list into a kylist.
unwrapped_np_concatenate = np.concatenate
def concatwrapper(*args, **kwargs):
    args = (kylist(*(args[0])),) + args[1:]
    return unwrapped_np_concatenate(*args, **kwargs)
np.concatenate = concatwrapper
