"""Convenience functions built on top of `make_vjp`."""
from __future__ import absolute_import
from functools import partial
import autograd.numpy as np
from autograd.core import make_vjp, getval, isnode, vspace
from .errors import add_error_hints
from collections import OrderedDict
from inspect import getargspec
import itertools as it
import warnings

def grad(fun, argnum=0):
    """
    Returns a function which computes the gradient of `fun` with respect to
    positional argument number `argnum`. The returned function takes the same
    arguments as `fun`, but returns the gradient instead. The function `fun`
    should be scalar-valued. The gradient has the same type as the argument."""

    def scalar_fun(*args, **kwargs):
        return as_scalar(fun(*args, **kwargs))

    @attach_name_and_doc(fun, argnum, 'Gradient')
    @add_error_hints
    def gradfun(*args,**kwargs):
        args = list(args)
        args[argnum] = safe_type(args[argnum])
        vjp, ans = make_vjp(scalar_fun, argnum)(*args, **kwargs)
        return vjp(cast_to_same_dtype(1.0, ans))

    return gradfun

def jacobian(fun, argnum=0):
    """
    Returns a function which computes the Jacobian of `fun` with respect to
    positional argument number `argnum`, which must be a scalar or array. Unlike
    `grad` it is not restricted to scalar-output functions, but also it cannot
    take derivatives with respect to some argument types (like lists or dicts).
    If the input to `fun` has shape (in1, in2, ...) and the output has shape
    (out1, out2, ...) then the Jacobian has shape (out1, out2, ..., in1, in2, ...).
    """
    def getshape(val):
        val = getval(val)
        assert np.isscalar(val) or isinstance(val, np.ndarray), \
            'Jacobian requires input and output to be scalar- or array-valued'
        return np.shape(val)

    def unit_vectors(shape):
        for idxs in it.product(*map(range, shape)):
            vect = np.zeros(shape)
            vect[idxs] = 1
            yield vect

    concatenate = lambda lst: np.concatenate(map(np.atleast_1d, lst))

    @attach_name_and_doc(fun, argnum, 'Jacobian')
    @add_error_hints
    def jacfun(*args, **kwargs):
        vjp, ans = make_vjp(fun, argnum)(*args, **kwargs)
        outshape = getshape(ans)
        grads = map(vjp, unit_vectors(outshape))
        jacobian_shape = outshape + getshape(args[argnum])
        return np.reshape(concatenate(grads), jacobian_shape)

    return jacfun

def grad_named(fun, argname):
    '''Takes gradients with respect to a named argument.
       Doesn't work on *args or **kwargs.'''
    arg_index = getargspec(fun).args.index(argname)
    return grad(fun, arg_index)

def multigrad(fun, argnums=[0]):
    """Takes gradients wrt multiple arguments simultaneously."""
    def combined_arg_fun(multi_arg, *args, **kwargs):
        extra_args_list = list(args)
        for argnum_ix, arg_ix in enumerate(argnums):
            extra_args_list[arg_ix] = multi_arg[argnum_ix]
        return fun(*extra_args_list, **kwargs)
    gradfun = grad(combined_arg_fun, argnum=0)
    def gradfun_rearranged(*args, **kwargs):
        multi_arg = tuple([args[i] for i in argnums])
        return gradfun(multi_arg, *args, **kwargs)
    return gradfun_rearranged

def elementwise_grad(fun, argnum=0):
    """Like `jacobian`, but produces a function which computes just the diagonal
    of the Jacobian, and does the computation in one pass rather than in a loop.
    Note: this is only valid if the Jacobian is diagonal. Only arrays are
    currently supported. Can be used for broadcasting."""
    def sum_output(*args, **kwargs):
        return np.sum(fun(*args, **kwargs))
    return grad(sum_output, argnum=argnum)

def hessian(fun, argnum=0):
    "Returns a function that computes the exact Hessian."
    return jacobian(jacobian(fun, argnum), argnum)

def hessian_vector_product(fun, argnum=0):
    """Builds a function that returns the exact Hessian-vector product.
    The returned function has arguments (*args, vector, **kwargs), and takes
    roughly 4x as long to evaluate as the original function."""
    fun_grad = grad(fun, argnum)
    def vector_dot_grad(*args, **kwargs):
        args, vector = args[:-1], args[-1]
        return np.tensordot(fun_grad(*args, **kwargs), vector, np.ndim(vector))
    return grad(vector_dot_grad, argnum)  # Grad wrt original input.

def vector_jacobian_product(fun, argnum=0):
    """Builds a function that returns the exact vector-Jacobian product, that
    is the Jacobian matrix left-multiplied by vector. The returned function
    has arguments (*args, vector, **kwargs)."""
    def vector_dot_fun(*args, **kwargs):
        args, vector = args[:-1], args[-1]
        return np.tensordot(vector, fun(*args, **kwargs), axes=np.ndim(vector))
    return jacobian(vector_dot_fun, argnum)  # Grad wrt original input.

def value_and_grad(fun, argnum=0):
    """Returns a function that returns both value and gradient. Suitable for use
    in scipy.optimize"""
    def double_val_fun(*args, **kwargs):
        val = fun(*args, **kwargs)
        return val, getval(val)
    gradval_and_val = grad_and_aux(double_val_fun, argnum)
    flip = lambda x, y: (y, x)
    return lambda *args, **kwargs: flip(*gradval_and_val(*args, **kwargs))

def grad_and_aux(fun, argnum=0):
    """Builds a function that returns the gradient of the first output and the
    (unmodified) second output of a function that returns two outputs."""
    def grad_and_aux_fun(*args, **kwargs):
        saved = lambda: None
        def return_val_save_aux(*args, **kwargs):
            val, saved.aux = fun(*args, **kwargs)
            return val
        gradval = grad(return_val_save_aux, argnum)(*args, **kwargs)
        return gradval, saved.aux

    return grad_and_aux_fun

def multigrad_dict(fun):
    "Takes gradients wrt all arguments simultaneously,"
    "returns a dict mapping 'argname' to 'gradval'"

    import funcsigs
    sig = funcsigs.signature(fun)

    def select(preds, lst):
        idx = lambda item: next(
            (i for i, pred in enumerate(preds) if pred(item)), len(preds))
        results = [[] for _ in preds] + [[]]
        for item in lst:
            results[idx(item)].append(item)
        return results

    is_var_pos = lambda name: sig.parameters[name].kind == sig.parameters[name].VAR_POSITIONAL
    is_var_kwd = lambda name: sig.parameters[name].kind == sig.parameters[name].VAR_KEYWORD
    var_pos, var_kwd, argnames = select([is_var_pos, is_var_kwd], sig.parameters)

    todict = lambda dct: {key:dct[key] for key in dct}

    def apply_defaults(arguments):
        defaults = {name: param.default for name, param in sig.parameters.items()
                    if param.default is not param.empty}
        return OrderedDict((name, arguments[name] if name in arguments else defaults[name])
                           for name in sig.parameters)

    def gradfun(*args, **kwargs):
        bindings = sig.bind(*args, **kwargs)

        args = lambda dct: tuple(dct[var_pos[0]]) if var_pos else ()
        kwargs = lambda dct: todict(dct[var_kwd[0]]) if var_kwd else {}
        others = lambda dct: tuple(dct[argname] for argname in argnames
                                   if argname not in var_kwd + var_pos)

        newfun = lambda dct: fun(*(others(dct) + args(dct)), **kwargs(dct))

        argdict = apply_defaults(bindings.arguments)
        grad_dict = grad(newfun)(dict(argdict))
        return OrderedDict((argname, grad_dict[argname]) for argname in argdict)

    return gradfun

def attach_name_and_doc(fun, argnum, opname):
    namestr = "{op}_{fun}_wrt_argnum_{argnum}".format(
        op=opname.lower(), fun=getattr(fun, '__name__', '[unknown name]'), argnum=argnum)
    docstr = "{op} of function {fun} with respect to argument number {argnum}. " \
        "Has the same arguments as {fun} but the return value has type of " \
        "argument {argnum}.".format(op=opname, fun=getattr(fun, '__name__', '[unknown name]'),
        argnum=argnum)

    def wrap(gradfun):
        try:
            gradfun.__name__ = namestr
            gradfun.__doc__ = docstr
        finally:
            return gradfun
    return wrap

def safe_type(value):
    if isinstance(value, int):
        warnings.warn("Casting int to float to handle differentiation.")
        return float(value)
    else:
        return value

def as_scalar(x):
    vs = vspace(getval(x))
    if vs.iscomplex:
        x = np.real(x)
    if vs.shape == ():
        return x
    elif vs.size == 1:
        return x.reshape(())
    else:
        raise TypeError(
            "Output {} can't be cast to float. "
            "Function grad requires a scalar-valued function. "
            "Try jacobian or elementwise_grad.".format(getval(x)))

def cast_to_same_dtype(value, example):
    if hasattr(example, 'dtype') and example.dtype.type is not np.float64:
        return np.array(value, dtype=example.dtype)
    else:
        return value
