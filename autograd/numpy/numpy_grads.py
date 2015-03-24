from autograd.core import primitive as P, Node, log, swap_args
import numpy as np_original
import operator as op
from . import numpy_wrapper as anp
from .numpy_extra import take, untake

# ----- ndarray binary operators -----

I = lambda x : x
anp.ndarray.__dict__['__neg__'].defgrad(lambda ans, x: op.neg)
anp.ndarray.__dict__['__add__'].defgrad(lambda ans, x, y : unbroadcast(ans, x, I))
anp.ndarray.__dict__['__add__'].defgrad(lambda ans, x, y : unbroadcast(ans, y, I), argnum=1)
anp.ndarray.__dict__['__mul__'].defgrad(lambda ans, x, y : unbroadcast(ans, x, lambda g : y * g))
anp.ndarray.__dict__['__mul__'].defgrad(lambda ans, x, y : unbroadcast(ans, y, lambda g : x * g), argnum=1)
anp.ndarray.__dict__['__sub__'].defgrad(lambda ans, x, y : unbroadcast(ans, x, I))
anp.ndarray.__dict__['__sub__'].defgrad(lambda ans, x, y : unbroadcast(ans, y, op.neg), argnum=1)
anp.ndarray.__dict__['__div__'].defgrad(lambda ans, x, y : unbroadcast(ans, x, lambda g : g / y))
anp.ndarray.__dict__['__div__'].defgrad(lambda ans, x, y : unbroadcast(ans, y, lambda g : - g * x / y**2), argnum=1)
anp.ndarray.__dict__['__pow__'].defgrad(lambda ans, x, y : unbroadcast(ans, x, lambda g : g * y * x ** (y - 1)))
anp.ndarray.__dict__['__pow__'].defgrad(lambda ans, x, y : unbroadcast(ans, y, lambda g : g * log(x) * x ** y), argnum=1)

anp.ndarray.__dict__['__radd__'].grads = swap_args(anp.ndarray.__dict__['__add__'].grads)
anp.ndarray.__dict__['__rmul__'].grads = swap_args(anp.ndarray.__dict__['__mul__'].grads)
anp.ndarray.__dict__['__rsub__'].grads = swap_args(anp.ndarray.__dict__['__sub__'].grads)
anp.ndarray.__dict__['__rdiv__'].grads = swap_args(anp.ndarray.__dict__['__div__'].grads)
anp.ndarray.__dict__['__rpow__'].grads = swap_args(anp.ndarray.__dict__['__pow__'].grads)

# ----- Simple grads -----

anp.abs.defgrad(   lambda ans, x : lambda g : anp.sign(x) * g)
anp.exp.defgrad(   lambda ans, x : lambda g : ans * g)
anp.log.defgrad(   lambda ans, x : lambda g : g / x)
anp.sin.defgrad(   lambda ans, x : lambda g : g * anp.cos(x))
anp.cos.defgrad(   lambda ans, x : lambda g : - g * anp.sin(x))
anp.tan.defgrad(   lambda ans, x : lambda g : g / anp.cos(x) **2)
anp.sinh.defgrad(  lambda ans, x : lambda g : g * anp.cosh(x))
anp.cosh.defgrad(  lambda ans, x : lambda g : g * anp.sinh(x))
anp.tanh.defgrad(  lambda ans, x : lambda g : g / anp.cosh(x) **2)
anp.square.defgrad(lambda ans, x : lambda g : g * 2 * x)
anp.sqrt.defgrad(  lambda ans, x : lambda g : g * 0.5 * x**-0.5)
anp.sign.defgrad(  lambda ans, x : lambda g : 0.0)
anp.reshape.defgrad( lambda ans, x, shape, order=None : lambda g : anp.reshape(g, x.shape, order=order))
anp.ravel.defgrad(   lambda ans, x, order=None    : lambda g : anp.reshape(g, x.shape, order=order))
anp.expand_dims.defgrad(lambda ans, x, axis : lambda g : anp.squeeze(g, axis))
anp.squeeze.defgrad(    lambda ans, x, axis : lambda g : anp.repeat(g, x.shape[axis], axis))
anp.repeat.defgrad(     lambda ans, x, shape, axis  : lambda g : anp.sum(g, axis, keepdims=True))
anp.transpose.defgrad(  lambda ans, x               : lambda g : anp.transpose(g))
anp.split.defgrad(      lambda ans, x, idxs, axis=0 : lambda g : anp.concatenate(g, axis=axis))
anp.diag.defgrad(       lambda ans, x               : lambda g : anp.diag(g))
anp.trace.defgrad(      lambda ans, x               : lambda g : g * anp.eye(x.shape[0]))
anp.full.defgrad(     lambda ans, shape, fill_value : lambda g : anp.sum(g), argnum=1)

# ----- Trickier grads -----

isarray = lambda x : isinstance(getval(x), anp.ndarray)
getval = lambda x : x.value if isinstance(x, Node) else x

def make_grad_np_sum(ans, x, axis=None, keepdims=False):
    if not isarray(x):
        return I
    shape = x.shape
    if axis is None:
        return lambda g : anp.full(shape, g)
    else:
        if keepdims:
            return lambda g : anp.repeat(g, shape[axis], axis)
        else:
            return lambda g : anp.repeat(anp.expand_dims(g, axis), shape[axis], axis)
anp.sum.defgrad(make_grad_np_sum)

def make_grad_np_mean(ans, x, axis=None, keepdims=False):
    if not isarray(x):
        return I
    shape = x.shape
    if axis is None:
        return lambda g : anp.full(shape, g) / anp.prod(shape)
    else:
        if keepdims:
            return lambda g : anp.repeat(g, shape[axis], axis) / shape[axis]
        else:
            return lambda g : anp.repeat(anp.expand_dims(g, axis), shape[axis], axis) / shape[axis]
anp.mean.defgrad(make_grad_np_mean)

def make_grad_np_max(ans, x):
    def gradfun(g):
        idxs = anp.argmax(getval(x))
        shape = x.shape
        return untake(g, anp.unravel_index(idxs, shape), shape)
    return gradfun
anp.max.defgrad(make_grad_np_max)

def make_grad_np_dot_A(ans, A, B):
    def grad_np_dot_A(g):
        if B.ndim is 2:
            return anp.dot(g, B.T)
        elif A.ndim is 2:
            return anp.outer(g, B)
        else:
            return g * B
    return grad_np_dot_A
anp.dot.defgrad(make_grad_np_dot_A)
def make_grad_np_dot_B(ans, A, B):
    def grad_np_dot_B(g):
        if A.ndim is 2:
            return anp.dot(A.T, g)
        elif B.ndim is 2:
            return anp.outer(A, g)
        else:
            return g * A
    return grad_np_dot_B
anp.dot.defgrad(make_grad_np_dot_B, argnum=1)

def make_grad_concatenate_args(argnum, ans, axis, *args):
    start = sum([a.shape[axis] for a in args[:argnum-1]])
    idxs = [slice(None)] * ans.ndim
    idxs[axis] = slice(start, start + args[argnum-1].shape[axis])
    return lambda g : take(g, idxs)
anp.concatenate_args.gradmaker = make_grad_concatenate_args

# ----- Handle broadcasting -----

def unbroadcast(ans, x, fun):
    if isarray(x):
        shape = x.shape
        def new_fun(g):
            result = fun(g)
            while result.ndim > len(shape):
                result = anp.sum(result, axis=0)
            for axis, size in enumerate(shape):
                if size is 1:
                    result = anp.sum(result, axis=axis, keepdims=True)
            assert result.shape == shape
            return result
    else:
        new_fun = lambda g : anp.sum(fun(g))
    new_fun.__name__ = "unbroadcast_{0}".format(fun.__name__)
    return new_fun
