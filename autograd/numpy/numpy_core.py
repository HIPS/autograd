from __future__ import absolute_import
from functools import partial
import numpy as np
from copy import copy
import operator as op
from autograd.core import primitive, Node, log, reverse_args

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

P = primitive
W = wrap_output

# Differentiable functions
abs    = P(np.abs)
exp    = P(np.exp)
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

# ----- Grads -----

isarray = lambda x : isinstance(getval(x), np.ndarray)
isfloat = lambda x : isinstance(getval(x), float)
getval = lambda x : x.value if isinstance(x, Node) else x

def unbroadcast(ans, x, y, funs):
    return [unbroadcast_fun(ans, x, funs[0]),
            unbroadcast_fun(ans, y, funs[1])]

def unbroadcast_fun(ans, x, fun):
    if isfloat(x):
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

I = lambda x : x
ndarray.__dict__['__add__'].gradmaker = lambda ans, x, y : unbroadcast(ans, x, y, [I, I])
ndarray.__dict__['__mul__'].gradmaker = lambda ans, x, y : unbroadcast(ans, x, y, [lambda g : y * g, lambda g : x * g])
ndarray.__dict__['__sub__'].gradmaker = lambda ans, x, y : unbroadcast(ans, x, y, [I, op.neg])
ndarray.__dict__['__div__'].gradmaker = lambda ans, x, y : unbroadcast(ans, x, y, [lambda g : g / y, lambda g : - g * x / y**2])
ndarray.__dict__['__pow__'].gradmaker = lambda ans, x, y : unbroadcast(ans, x, y, [lambda g : g * y * x ** (y - 1),
                                                                                   lambda g : g * log(x) * x ** y])

ndarray.__dict__['__radd__'].gradmaker = reverse_args(ndarray.__dict__['__add__'].gradmaker)
ndarray.__dict__['__rmul__'].gradmaker = reverse_args(ndarray.__dict__['__mul__'].gradmaker)
ndarray.__dict__['__rsub__'].gradmaker = reverse_args(ndarray.__dict__['__sub__'].gradmaker)
ndarray.__dict__['__rdiv__'].gradmaker = reverse_args(ndarray.__dict__['__div__'].gradmaker)
ndarray.__dict__['__rpow__'].gradmaker = reverse_args(ndarray.__dict__['__pow__'].gradmaker)

abs.gradmaker    = lambda ans, x : [lambda g : sign(x) * g]
exp.gradmaker    = lambda ans, x : [lambda g : ans * g]
sin.gradmaker    = lambda ans, x : [lambda g : g * cos(x)]
cos.gradmaker    = lambda ans, x : [lambda g : - g * sin(x)]
tan.gradmaker    = lambda ans, x : [lambda g : g / cos(x) **2]
sinh.gradmaker   = lambda ans, x : [lambda g : g * cosh(x)]
cosh.gradmaker   = lambda ans, x : [lambda g : g * sinh(x)]
tanh.gradmaker   = lambda ans, x : [lambda g : g / cosh(x) **2]
square.gradmaker = lambda ans, x : [lambda g : g * 2 * x]
sqrt.gradmaker   = lambda ans, x : [lambda g : g * 0.5 * x**-0.5]
sign.gradmaker   = lambda ans, x : [lambda g : 0.0]
full.gradmaker   = lambda ans, shape, fill_value : [None, lambda g :  sum(g)]
reshape.gradmaker  = lambda ans, x, shape, order=None : [lambda g : reshape(g, x.shape, order=order)]
ravel.gradmaker    = lambda ans, x, order=None    : [lambda g : reshape(g, x.shape, order=order)]
expand_dims.gradmaker = lambda ans, x, axis : [lambda g : squeeze(g, axis)]
squeeze.gradmaker     = lambda ans, x, axis : [lambda g : repeat(g, x.shape[axis], axis)]
repeat.gradmaker      = lambda ans, x, shape, axis  : [lambda g : sum(g, axis, keepdims=True)]
transpose.gradmaker   = lambda ans, x               : [lambda g : transpose(g)]
split.gradmaker       = lambda ans, x, idxs, axis=0 : [lambda g : concatenate(g, axis=axis)]
diag.gradmaker        = lambda ans, x               : [lambda g : diag(g)]
trace.gradmaker       = lambda ans, x               : [lambda g : g * eye(x.shape[0])]

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
sum.gradmaker = make_grad_np_sum

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
mean.gradmaker = make_grad_np_mean

def make_grad_np_max(ans, x):
    def gradfun(g):
        idxs = argmax(getval(x))
        shape = x.shape
        return untake(g, unravel_index(idxs, shape), shape)
    return [gradfun]
max.gradmaker = make_grad_np_max

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
dot.gradmaker = make_grad_np_dot

take = P(lambda A, idx : A[idx])
def make_grad_take(ans, A, idx):
    shape = A.shape
    return [lambda g : untake(g, idx, shape)]
take.gradmaker =  make_grad_take

untake = P(lambda x, idx, shape : SparseArray(shape, idx, x))
untake.gradmaker = lambda ans, x, idx, shape : [lambda g : take(g, idx)]

# ----- Subtler gradients -----

concatenate_orig = np.concatenate
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
concatenate_args = P(W(concatenate_args))
concatenate_args.gradmaker = make_grad_concatenate_args
concatenate = lambda arr_list, axis=0 : concatenate_args(axis, *arr_list)

# ----- Node version of ndarray -----

class ArrayNode(Node):
    __slots__ = []
    __getitem__ = take
    # Constants w.r.t float data just pass though
    @property
    def shape(self): return self.value.shape
    @property
    def ndim(self): return self.value.ndim
    @property
    def size(self): return self.value.size

    # Differentiable unary methods just apply to self
    squeeze = squeeze
    ravel   = ravel
    reshape = reshape
    sum     = sum
    mean    = mean
    @property
    def T(self): return transpose(self)
    __neg__ = P(op.neg)

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

ArrayNode.__dict__['__neg__'].gradmaker = lambda ans, x : [op.neg]

# ----- Sparse array -----

class SparseArray(object):
    __array_priority__ = 150.0
    def __init__(self, shape, idx, val):
        self.shape = shape
        self.idx = idx
        self.val = val

    def __add__(self, other):
        array = zeros(self.shape) if other is 0 else copy(other)
        array[self.idx] += self.val
        return array

    def __radd__(self, other):
        return self.__add__(other)

class SparseArrayNode(Node):
    __slots__ = []
    __add__  = P(SparseArray.__add__)
    __radd__ = P(SparseArray.__radd__)
Node.type_mappings[SparseArray] = SparseArrayNode

SparseArrayNode.__dict__['__add__'].gradmaker  = lambda ans, x, y : unbroadcast(ans, x, y, [I, I])
SparseArrayNode.__dict__['__radd__'].gradmaker = lambda ans, x, y : unbroadcast(ans, x, y, [I, I])
