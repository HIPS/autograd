from __future__ import absolute_import
from functools import partial
from copy import copy
from numpy import *
import numpy as np_orig
import operator as op
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

def unbroadcast(ans, x, y, funs):
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
                    result = sum(result, axis=axis, keepdims=True)
            assert result.shape == shape
            return result
        return new_fun
    else:
        return fun

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

# Numpy doesn't support keepdims for subclasses so this is the workaround
def keep_keepdims(fun, funname):
    def new_fun(*args, **kwargs):
        x = args[0]
        return getattr(x, funname)(*args[1:], **kwargs) if isinstance(x, np_orig.ndarray) else x
    return new_fun
sum = primitive(keep_keepdims(sum, 'sum'), make_grad_np_sum)

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
mean = primitive(keep_keepdims(mean, 'mean'), make_grad_np_mean)

# ----- Slightly modified version of ndarray -----

P = primitive
I = lambda x : x # Identity operator

grad_neg = lambda ans, x    : [op.neg]
grad_add = lambda ans, x, y : unbroadcast(ans, x, y, [I, I])
grad_mul = lambda ans, x, y : unbroadcast(ans, x, y, [lambda g : y * g, lambda g : x * g])
grad_sub = lambda ans, x, y : unbroadcast(ans, x, y, [I, op.neg])
grad_div = lambda ans, x, y : unbroadcast(ans, x, y, [lambda g : g / y, lambda g : - g * x / y**2])
grad_pow = lambda ans, x, y : unbroadcast(ans, x, y, [lambda g : g * y * x ** (y - 1),
                                                      lambda g : g * log(x) * x ** y])
grad_log = lambda ans, x    : [lambda g : g / x]

def reverse_args(fun):
    def new_fun(ans, x, y):
        return fun(ans, y, x)[::-1]
    return new_fun

def make_grad_take(ans, A, idx):
    shape = A.shape
    return [lambda g : untake(g, idx, shape)]

class ndarray(np_orig.ndarray):
    # Wrap binary ops since the other operand could be a Node
    __add__  = P(np_orig.ndarray.__add__ , grad_add)
    __sub__  = P(np_orig.ndarray.__sub__,  grad_sub)
    __mul__  = P(np_orig.ndarray.__mul__,  grad_mul)
    __pow__  = P(np_orig.ndarray.__pow__,  grad_pow)
    __div__  = P(np_orig.ndarray.__div__,  grad_div)
    __radd__ = P(np_orig.ndarray.__radd__, reverse_args(grad_add))
    __rsub__ = P(np_orig.ndarray.__rsub__, reverse_args(grad_sub))
    __rmul__ = P(np_orig.ndarray.__rmul__, reverse_args(grad_mul))
    __rpow__ = P(np_orig.ndarray.__rpow__, reverse_args(grad_pow))
    __rdiv__ = P(np_orig.ndarray.__rdiv__, reverse_args(grad_div))

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

def wrap_output(fun):
    def wrapped_fun(*args, **kwargs):
        ans = fun(*args, **kwargs)
        if isinstance(ans, np_orig.ndarray):
            ans = ans.view(ndarray)
        return ans
    return wrapped_fun
zeros = wrap_output(zeros)
ones = wrap_output(ones)

# ----- Numpy gradients -----

W = wrap_output
P = primitive
isarray = lambda x : isinstance(getval(x), ndarray)
I = lambda x : x

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
full   = P(W(full),   lambda ans, shape, fill_value : [None, lambda g :  sum(g)])
reshape  = P(reshape, lambda ans, x, shape, order=None : [lambda g : reshape(g, x.shape, order=order)])
ravel    = P(W(ravel), lambda ans, x, order=None    : [lambda g : reshape(g, x.shape, order=order)])
expand_dims = P(W(expand_dims), lambda ans, x, axis : [lambda g : squeeze(g, axis)])
squeeze     = P(squeeze,        lambda ans, x, axis : [lambda g : repeat(g, x.shape[axis], axis)])
repeat      = P(repeat,      lambda ans, x, shape, axis  : [lambda g : sum(g, axis, keepdims=True)])
transpose   = P(transpose,   lambda ans, x               : [lambda g : transpose(g)])
split       = P(split,       lambda ans, x, idxs, axis=0 : [lambda g : concatenate(g, axis=axis)])
diag        = P(W(diag),     lambda ans, x               : [lambda g : diag(g)])
trace       = P(trace,       lambda ans, x               : [lambda g : g * eye(x.shape[0])])

# ----- Subtler gradients -----

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
concatenate_args = P(W(concatenate_args), make_grad_concatenate_args)
concatenate = lambda arr_list, axis=0 : concatenate_args(axis, *arr_list)

# ----- Node version of ndarray -----

class ArrayNode(Node):
    # Constants w.r.t float data just pass though
    @property
    def shape(self): return self.value.shape
    @property
    def ndim(self): return self.value.ndim
    @property
    def size(self): return self.value.size

    # Differentiable unary methods just apply to self
    squeeze = squeeze
    ravel = ravel
    reshape = reshape
    sum = sum
    mean = mean

    @property
    def T(self): return transpose(self)
    __neg__ = P(op.neg,  grad_neg)

    # Binary ops already wrapped by autograd.numpy.ndarray
    __add__  = ndarray.__add__.__func__
    __sub__  = ndarray.__sub__.__func__
    __mul__  = ndarray.__mul__.__func__
    __pow__  = ndarray.__pow__.__func__
    __div__  = ndarray.__div__.__func__
    __radd__ = ndarray.__radd__.__func__
    __rsub__ = ndarray.__rsub__.__func__
    __rmul__ = ndarray.__rmul__.__func__
    __rpow__ = ndarray.__rpow__.__func__
    __rdiv__ = ndarray.__rdiv__.__func__
Node.type_mappings[ndarray] = ArrayNode
