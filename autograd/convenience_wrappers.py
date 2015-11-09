"""Convenience functions built on top of `grad`."""
from __future__ import absolute_import
import autograd.numpy as np
from autograd.core import grad, getval, jacobian
from collections import OrderedDict


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
        return np.dot(vector, fun_grad(*args, **kwargs))
    return grad(vector_dot_grad, argnum)  # Grad wrt original input.

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
