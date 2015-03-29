from autograd.core import log, swap_args, getval, nondifferentiable_ops

import operator as op
from . import numpy_wrapper as anp
from .numpy_extra import take

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

for comp_op in nondifferentiable_ops:
    anp.ndarray.__dict__[comp_op].defgrad_is_zero()
    anp.ndarray.__dict__[comp_op].defgrad_is_zero(argnum=1)

anp.ndarray.__dict__['__radd__'].grads = swap_args(anp.ndarray.__dict__['__add__'].grads)
anp.ndarray.__dict__['__rmul__'].grads = swap_args(anp.ndarray.__dict__['__mul__'].grads)
anp.ndarray.__dict__['__rsub__'].grads = swap_args(anp.ndarray.__dict__['__sub__'].grads)
anp.ndarray.__dict__['__rdiv__'].grads = swap_args(anp.ndarray.__dict__['__div__'].grads)
anp.ndarray.__dict__['__rpow__'].grads = swap_args(anp.ndarray.__dict__['__pow__'].grads)

# ----- Functions that are constant w.r.t. continuous inputs -----

anp.floor.defgrad_is_zero()
anp.ceil.defgrad_is_zero()
anp.round.defgrad_is_zero()
anp.rint.defgrad_is_zero()
anp.around.defgrad_is_zero()
anp.all.defgrad_is_zero()
anp.any.defgrad_is_zero()
anp.argmax.defgrad_is_zero()
anp.argmin.defgrad_is_zero()
anp.argpartition.defgrad_is_zero()
anp.argsort.defgrad_is_zero()
anp.nonzero.defgrad_is_zero()
anp.searchsorted.defgrad_is_zero()
anp.sign.defgrad_is_zero()
anp.ndim.defgrad_is_zero()

# ----- Simple grads -----

anp.abs.defgrad(     lambda ans, x : lambda g : anp.sign(x) * g)
anp.fabs.defgrad(    lambda ans, x : lambda g : anp.sign(x) * g)
anp.absolute.defgrad(lambda ans, x : lambda g : anp.sign(x) * g)
anp.power.defgrad( lambda ans, x, y : lambda g : g * y * x ** (y - 1))
anp.power.defgrad( lambda ans, x, y : lambda g : g * log(x) * x ** y, argnum=1)
anp.mod.defgrad(      lambda ans, x, y : lambda g : g)
anp.fmod.defgrad(     lambda ans, x, y : lambda g : g)
anp.remainder.defgrad(lambda ans, x, y : lambda g : g)
anp.mod.defgrad(      lambda ans, x, y : lambda g : -g * anp.floor(x/y), argnum=1)
anp.fmod.defgrad(     lambda ans, x, y : lambda g : -g * anp.floor(x/y), argnum=1)
anp.remainder.defgrad(lambda ans, x, y : lambda g : -g * anp.floor(x/y), argnum=1)
anp.exp.defgrad(   lambda ans, x : lambda g : ans * g)
anp.log.defgrad(   lambda ans, x : lambda g : g / x)
anp.sin.defgrad(   lambda ans, x : lambda g : g * anp.cos(x))
anp.cos.defgrad(   lambda ans, x : lambda g : - g * anp.sin(x))
anp.tan.defgrad(   lambda ans, x : lambda g : g / anp.cos(x) **2)
anp.sinh.defgrad(  lambda ans, x : lambda g : g * anp.cosh(x))
anp.cosh.defgrad(  lambda ans, x : lambda g : g * anp.sinh(x))
anp.tanh.defgrad(  lambda ans, x : lambda g : g / anp.cosh(x) **2)
anp.arcsinh.defgrad(lambda ans, x : lambda g : g / anp.sqrt(x**2 + 1))
anp.arccosh.defgrad(lambda ans, x : lambda g : g / anp.sqrt(x**2 - 1))
anp.arctanh.defgrad(lambda ans, x : lambda g : g / (1 - x**2))
anp.square.defgrad(lambda ans, x : lambda g : g * 2 * x)
anp.sqrt.defgrad(  lambda ans, x : lambda g : g * 0.5 * x**-0.5)
anp.reshape.defgrad( lambda ans, x, shape, order=None : lambda g : anp.reshape(g, x.shape, order=order))
anp.roll.defgrad(    lambda ans, x, shift, axis=None  : lambda g : anp.roll(g, -shift, axis=axis))
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

def repeat_to_match_shape(x, axis, keepdims):
    """Returns a function that repeats an array along axis to get a given shape.
       Also returns the number of repetitions of the array."""
    if not isarray(x):
        return I, 1
    shape = x.shape
    if axis is None:
        return lambda g : anp.full(shape, g), anp.prod(shape)
    else:
        if keepdims:
            return lambda g : anp.repeat(g, shape[axis], axis), shape[axis]
        else:
            return lambda g : anp.repeat(anp.expand_dims(g, axis),
                                         shape[axis], axis), shape[axis]

def make_grad_np_sum(ans, x, axis=None, keepdims=False):
    repeater, _ = repeat_to_match_shape(x, axis, keepdims)
    return repeater
anp.sum.defgrad(make_grad_np_sum)

def make_grad_np_mean(ans, x, axis=None, keepdims=False):
    repeater, num_reps = repeat_to_match_shape(x, axis, keepdims)
    return lambda g: repeater(g) / num_reps
anp.mean.defgrad(make_grad_np_mean)

def make_grad_chooser(ans, x, axis=None, keepdims=None):
    """Builds gradient of functions that choose a single item, such as min or max."""
    repeater, _ = repeat_to_match_shape(x, axis, keepdims)
    argmax_locations = x == repeater(ans)
    return lambda g: repeater(g) * argmax_locations
anp.max.defgrad(make_grad_chooser)
anp.min.defgrad(make_grad_chooser)
anp.amax.defgrad(make_grad_chooser)
anp.amin.defgrad(make_grad_chooser)

def reverse_axis(x, axis):
    x = x.swapaxes(axis, 0)
    x = x[::-1,...]
    return x.swapaxes(0, axis)

def make_grad_np_cumsum(ans, x, axis=None):
    if axis:
        return lambda g: reverse_axis(anp.cumsum(reverse_axis(g, axis)), axis)
    else:
        shape = x.shape
        return lambda g: anp.reshape(anp.cumsum(g[::-1], axis)[::-1], shape)
anp.cumsum.defgrad(make_grad_np_cumsum)

def make_grad_np_dot_A(ans, A, B):
    def grad_np_dot_A(g):
        if anp.ndim(A) is 0:
            return anp.sum(g * B)
        elif anp.ndim(B) is 2:
            return anp.dot(g, B.T)
        elif anp.ndim(A) is 2:
            if anp.ndim(B) is 0:
                return anp.dot(B, g)
            else:
                return anp.outer(g, B)
        else:
            return B * g
    return grad_np_dot_A
anp.dot.defgrad(make_grad_np_dot_A)
def make_grad_np_dot_B(ans, A, B):
    def grad_np_dot_B(g):
        if anp.ndim(B) is 0:
            return anp.sum(g * A)
        elif anp.ndim(A) is 2:
            return anp.dot(A.T, g)
        elif anp.ndim(B) is 2:
            if anp.ndim(A) is 0:
                return anp.dot(A, g)
            else:
                return anp.outer(A, g)
        else:
            return A * g
    return grad_np_dot_B
anp.dot.defgrad(make_grad_np_dot_B, argnum=1)

anp.outer.defgrad(lambda ans, a, b : lambda g : anp.dot(g, b.T))
anp.outer.defgrad(lambda ans, a, b : lambda g : anp.dot(a.T, g), argnum=1)

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
