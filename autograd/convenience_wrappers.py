"""Convenience functions built on top of `grad`."""
from __future__ import absolute_import
import autograd.numpy as np
from autograd.core import grad, getval, jacobian
from collections import OrderedDict
from future.utils import iteritems


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

def multigrad_dict(fun):
    "Takes gradients wrt all arguments simultaneously,"
    "returns a dict mapping 'argname' to 'gradval'"

    import funcsigs
    sig = funcsigs.signature(fun)

    def gradfun(*args, **kwargs):
        bindings = sig.bind(*args, **kwargs)
        var_positional = next((name for name, parameter in sig.parameters.items()
                               if parameter.kind == parameter.VAR_POSITIONAL), None)
        var_keyword = next((name for name, parameter in sig.parameters.items()
                            if parameter.kind == parameter.VAR_KEYWORD), None)
        args = lambda dct: dct[var_positional] if var_positional else ()
        kwargs = lambda dct: \
            dict(((key, dct[var_keyword][key]) for key in dct[var_keyword]) if var_keyword else (),
                 **{argname: dct[argname] for argname, parameter in sig.parameters.items()
                    if parameter.kind != parameter.VAR_POSITIONAL})

        newfun = lambda dct: fun(*args(dct), **kwargs(dct))
        grad_dict = grad(newfun)(dict(bindings.arguments))
        return OrderedDict((argname, grad_dict[argname]) for argname in bindings.arguments)

    return gradfun

def grad_and_aux(fun, argnum=0):
    """Builds a function that returns the gradient of the first output and the
    (unmodified) second output of a function that returns two outputs."""
    def grad_and_aux_fun(*args, **kwargs):
        saved_aux = []
        def return_val_save_aux(*args, **kwargs):
            val, aux = fun(*args, **kwargs)
            saved_aux.append(aux)
            return val
        gradval = grad(return_val_save_aux, argnum)(*args, **kwargs)
        return gradval, saved_aux[0]

    return grad_and_aux_fun

def value_and_grad(fun, argnum=0):
    """Returns a function that returns both value and gradient. Suitable for use
    in scipy.optimize"""
    def double_val_fun(*args, **kwargs):
        val = fun(*args, **kwargs)
        return val, getval(val)
    gradval_and_val = grad_and_aux(double_val_fun, argnum)

    def value_and_grad_fun(*args, **kwargs):
        gradval, val = gradval_and_val(*args, **kwargs)
        return val, gradval

    return value_and_grad_fun

def elementwise_grad(fun, argnum=0):
    """Like `jacobian`, but produces a function which computes just the diagonal
    of the Jacobian, and does the computation in one pass rather than in a loop.
    Note: this is only valid if the Jacobian is diagonal. Only arrays are
    currently supported."""
    def sum_output(*args, **kwargs):
        return np.sum(fun(*args, **kwargs))
    return grad(sum_output, argnum=argnum)

def hessian_vector_product(fun, argnum=0):
    """Builds a function that returns the exact Hessian-vector product.
    The returned function has arguments (*args, vector, **kwargs), and takes
    roughly 4x as long to evaluate as the original function."""
    fun_grad = grad(fun, argnum)
    def vector_dot_grad(*args, **kwargs):
        args, vector = args[:-1], args[-1]
        return np.dot(vector, fun_grad(*args, **kwargs))
    return grad(vector_dot_grad, argnum)  # Grad wrt original input.

def hessian(fun, argnum=0):
    "Returns a function that computes the exact Hessian."
    return jacobian(jacobian(fun, argnum), argnum)
