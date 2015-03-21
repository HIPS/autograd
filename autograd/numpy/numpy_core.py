from __future__ import absolute_import
from functools import partial
from copy import copy
from numpy import *
import numpy as np_orig
from autograd.core import primitive, Node, log, getval

# ----- Objects in numpy.__dict__ not imported by * -----

int     = np_orig.int
unicode = np_orig.unicode
complex = np_orig.complex
long    = np_orig.long
abs     = np_orig.abs
bool    = np_orig.bool
float   = np_orig.float
max     = np_orig.max
object  = np_orig.object
min     = np_orig.min
str     = np_orig.str
round   = np_orig.round

# ----- Broadcasting logic -----

isarray = lambda x : isinstance(getval(x), ndarray)
isfloat = lambda x : isinstance(getval(x), float)

def do_unbroadcast(ans, x, y, funs):
    return [unbroadcast_fun(ans, x, funs[0]),
            unbroadcast_fun(ans, y, funs[1])]

def unbroadcast_fun(ans, x, fun):
    if isfloat(x) and isarray(ans):
        return lambda g : sum(fun(g))
    elif isarray(x):
        shape = x.shape
        def new_fun(g):
            result = fun(g)
            while result.ndim > len(shape):
                result = sum(result, axis=0)
            for axis, size in enumerate(shape):
                if size is 1:
                    result = sum(result, axis, keepdims=True)
            return result
        return new_fun
    else:
        return fun

# ----- Node version of ndarray -----

class ArrayNode(Node):
    def __getitem__(self, idx):
        return take(self, idx)
    def reshape(self, shape, order=None):
        return reshape(self, shape, order=order)
    def ravel(self, order=None):
        return ravel(self, order=order)
    def squeeze(self, axis=None):
        return squeeze(self, axis=axis)
    def sum(self):
        return sum(self)
    @property
    def T(self): return transpose(self)
    @property
    def shape(self): return self.value.shape
    @property
    def ndim(self): return self.value.ndim
    @property
    def size(self): return self.value.size
Node.type_mappings[ndarray] = ArrayNode

# ----- Sparse array -----

class SparseArray(object):
    __array_priority__ = 50.0
    def __init__(self, shape, idx, val):
        self.shape = shape
        self.idx = idx
        self.val = val

    def __add__(self, other):
        array = zeros(self.shape) if other is 0 else copy(other)
        array[self.idx] += self.val
        return array

    def __radd__(self, other):
        return self + other

class SparseArrayNode(Node):
    pass
Node.type_mappings[SparseArray] = SparseArrayNode

take = lambda A, idx : A[idx]
def make_grad_take(ans, A, idx):
    shape = A.shape
    return [lambda g : untake(g, idx, shape)]
take = primitive(take, make_grad_take)

untake = lambda x, idx, shape : SparseArray(shape, idx, x)
untake = primitive(untake, lambda ans, x, idx, shape : [lambda g : take(g, idx)])

# ----- Numpy gradients -----

P = primitive
isarray = lambda x : isinstance(getval(x), ndarray)
I = lambda x : x # Identity operator

abs    = P(abs,    lambda ans, x : [lambda g : sign(x) * g])
exp    = P(exp,    lambda ans, x : [lambda g : ans * g])
sin    = P(sin,    lambda ans, x : [lambda g : g * cos(x)])
cos    = P(cos,    lambda ans, x : [lambda g : - g * sin(x)])
tan    = P(tan,    lambda ans, x : [lambda g : g / cos(x) **2])
sinh   = P(sinh,   lambda ans, x : [lambda g : g * cosh(x)])
cosh   = P(cosh,   lambda ans, x : [lambda g : g * sinh(x)])
tanh   = P(tanh,   lambda ans, x : [lambda g : g / cosh(x) **2])
square = P(square, lambda ans, x : [lambda g : g * 2 * x])
sqrt   = P(sqrt,   lambda ans, x : [lambda g : g * 0.5 * x**-0.5])
sign   = P(sign,   lambda ans, x : [lambda g : 0.0])
full   = P(full,   lambda ans, shape, fill_value : [None, lambda g :  sum(g)])
reshape     = P(reshape,     lambda ans, x, shape, order=None : [lambda g : reshape(g, x.shape, order=order)])
ravel       = P(ravel,       lambda ans, x, order=None   : [lambda g : reshape(g, x.shape, order=order)])
expand_dims = P(expand_dims, lambda ans, x, axis         : [lambda g : squeeze(g, axis)])
squeeze     = P(squeeze,     lambda ans, x, axis         : [lambda g : repeat(g, x.shape[axis], axis)])
repeat      = P(repeat,      lambda ans, x, shape, axis  : [lambda g : sum(g, axis, keepdims=True)])
transpose   = P(transpose,   lambda ans, x               : [lambda g : transpose(g)])
split       = P(split,       lambda ans, x, idxs, axis=0 : [lambda g : concatenate(g, axis=axis)])
diag        = P(diag,        lambda ans, x               : [lambda g : diag(g)])
trace       = P(trace,       lambda ans, x               : [lambda g : g * eye(x.shape[0])])

# ----- Subtler gradients -----

def make_grad_np_sum(ans, x, axis=None, keepdims=False):
    if not isarray(x):
        return [I]
    shape = x.shape
    if axis is None:
        return [lambda g : full(shape, g)]
    else:
        if keepdims:
            return [lambda g : repeat(g, shape[axis], axis)]
        else:
            return [lambda g : repeat(expand_dims(g, axis),
                                      shape[axis], axis)]
sum = P(sum, make_grad_np_sum)

def make_grad_np_mean(ans, x, axis=None, keepdims=False):
    if not isarray(x):
        return [I]
    shape = x.shape
    if axis is None:
        return [lambda g : full(shape, g) / prod(shape)]
    else:
        if keepdims:
            return [lambda g : repeat(g, shape[axis], axis) / shape[axis]]
        else:
            return [lambda g : repeat(expand_dims(g, axis),
                                      shape[axis], axis) / shape[axis]]
mean = P(mean, make_grad_np_mean)

def make_grad_np_max(ans, x):
    def gradfun(g):
        idxs = argmax(getval(x))
        shape = x.shape
        return untake(g, unravel_index(idxs, shape), shape)
    return [gradfun]
max = P(max, make_grad_np_max)

def make_grad_np_dot(ans, A, B):
    def grad_np_dot_A(g):
        if B.ndim is 2:
            return dot(g, B.T)
        elif A.ndim is 2:
            return outer(g, B)
        else:
            return g * B
    def grad_np_dot_B(g):
        if A.ndim is 2:
            return dot(A.T, g)
        elif B.ndim is 2:
            return outer(A, g)
        else:
            return g * A
    return [grad_np_dot_A, grad_np_dot_B]
dot = P(dot, make_grad_np_dot)

concatenate_orig = concatenate
def concatenate_args(axis, *args):
    return concatenate_orig(args, axis)
def make_grad_concatenate_args(ans, axis, *args):
    high = 0
    gradfuns = [None]
    for a in args:
        low = high
        high += a.shape[axis]
        idxs = [slice(None)] * ans.ndim
        idxs[axis] = slice(low, high)
        gradfuns.append(partial(take, idx=idxs))
    return gradfuns
concatenate_args = P(concatenate_args, make_grad_concatenate_args)
concatenate = lambda arr_list, axis=0 : concatenate_args(axis, *arr_list)

def make_grad_np_concatenate(ans, arr_list, axis=0):
    idxs = cumsum([a.shape[axis] for a in arr_list[:-1]])
    def grad_np_concatenate(g):
        return split(g, idxs, axis=axis)
    return [grad_np_concatenate]
concatenate = P(concatenate, make_grad_np_concatenate)
