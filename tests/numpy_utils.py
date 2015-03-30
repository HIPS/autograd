import itertools as it
import autograd.numpy.random as npr
from autograd import grad
from test_util import check_equivalent, check_grads, to_scalar

def combo_check(fun, argnums, *args, **kwargs):
    # Tests all combinations of args given.
    args = list(args)
    kwarg_key_vals = [[(key, val) for val in kwargs[key]] for key in kwargs]
    num_args = len(args)
    for args_and_kwargs in it.product(*(args + kwarg_key_vals)):
        cur_args = args_and_kwargs[:num_args]
        cur_kwargs = dict(args_and_kwargs[num_args:])
        check_fun_and_grads(fun, cur_args, cur_kwargs, argnums=argnums)

def check_fun_and_grads(fun, args, kwargs, argnums):
    try:
        # Confirm the forward pass itself is fine
        check_equivalent(fun(*args, **kwargs),
                         fun.fun(*args, **kwargs))

        # Check first derivatives
        def scalar_fun(*new_args):
            full_args = list(args)
            for i, argnum in enumerate(argnums):
                full_args[argnum] = new_args[i]
            return to_scalar(fun(*full_args, **kwargs))

        wrt_args = [args[i] for i in argnums]
        check_grads(scalar_fun, *wrt_args)

        # Check the second derivatives
        for i in range(len(argnums)):
            check_grads(grad(scalar_fun, argnum=i))
    except:
        print "Test failed! Args were", args, kwargs
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

def unary_ufunc_check(fun, lims=[-2, 2]):
    transform = lambda x : x * (lims[1] - lims[0]) + lims[0]
    scalar = transform(0.4)
    vector = transform(npr.rand(2))
    mat    = transform(npr.rand(3, 2))
    mat2   = transform(npr.rand(1, 2))
    combo_check(fun, (0,), [scalar, vector, mat, mat2])

def binary_ufunc_check(fun):
    scalar = 2.0
    vector = npr.randn(2)
    mat = npr.randn(3, 2)
    mat2 = npr.randn(1, 2)
    combo_check(fun, (0, 1), [scalar, vector, mat, mat2],
                             [scalar, vector, mat, mat2])
