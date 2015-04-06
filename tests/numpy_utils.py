import itertools as it
import autograd.numpy.random as npr
from autograd import grad, primitive
from autograd.util import check_equivalent, check_grads, to_scalar

def combo_check(fun, argnums, *args, **kwargs):
    # Tests all combinations of args given.
    args = list(args)
    kwarg_key_vals = [[(key, val) for val in kwargs[key]] for key in kwargs]
    num_args = len(args)
    for args_and_kwargs in it.product(*(args + kwarg_key_vals)):
        cur_args = args_and_kwargs[:num_args]
        cur_kwargs = dict(args_and_kwargs[num_args:])
        check_fun_and_grads(fun, cur_args, cur_kwargs, argnums=argnums)
        print ".",

def check_fun_and_grads(fun, args, kwargs, argnums):
    wrt_args = [args[i] for i in argnums]
    try:
        if isinstance(fun, primitive):
            check_equivalent(fun(*args, **kwargs),
                             fun.fun(*args, **kwargs))
    except:
        print "Value test failed! Args were", args, kwargs
        raise

    try:
        def scalar_fun(*new_args):
            full_args = list(args)
            for i, argnum in enumerate(argnums):
                full_args[argnum] = new_args[i]
            return to_scalar(fun(*full_args, **kwargs))
        check_grads(scalar_fun, *wrt_args)
    except:
        print "First derivative test failed! Args were", args, kwargs
        raise

    try:
        for i in range(len(argnums)):
            def d_scalar_fun(*args):
                return to_scalar(grad(scalar_fun, argnum=i)(*args))
            check_grads(d_scalar_fun, *wrt_args)
    except:
        print "Second derivative test failed! Args were", args, kwargs
        raise

def stat_check(fun):
    # Tests functions that compute statistics, like sum, mean, etc
    x = 3.5
    A = npr.randn()
    B = npr.randn(3)
    C = npr.randn(2, 3)
    D = npr.randn(1, 3)
    combo_check(fun, (0,), [x, A])
    combo_check(fun, (0,), [B, C, D], axis=[None, 0], keepdims=[True, False])
    combo_check(fun, (0,), [C, D], axis=[None, 0, 1], keepdims=[True, False])

def unary_ufunc_check(fun, lims=[-2, 2], **kwargs):
    scalar_int = transform(lims, 1)
    scalar = transform(lims, 0.4)
    vector = transform(lims, npr.rand(2))
    mat    = transform(lims, npr.rand(3, 2))
    mat2   = transform(lims, npr.rand(1, 2))
    combo_check(fun, (0,), [scalar_int, scalar, vector, mat, mat2], **kwargs)

def binary_ufunc_check(fun, lims_A=[-2, 2], lims_B=[-2, 2]):
    T_A = lambda x : transform(lims_A, x)
    T_B = lambda x : transform(lims_B, x)
    scalar_int = 1
    scalar = 0.6
    vector = npr.rand(2)
    mat    = npr.rand(3, 2)
    mat2   = npr.rand(1, 2)
    combo_check(fun, (0, 1), [T_A(scalar), T_A(scalar_int), T_A(vector), T_A(mat), T_A(mat2)],
                             [T_B(scalar), T_B(scalar_int), T_B(vector), T_B(mat), T_B(mat2)])

def transform(lims, x):
    return x * (lims[1] - lims[0]) + lims[0]
