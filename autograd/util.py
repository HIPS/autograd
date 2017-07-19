from __future__ import absolute_import
from __future__ import print_function
from copy import copy
from builtins import range

import autograd.numpy as np
from autograd.convenience_wrappers import grad
from autograd.core import vspace, vspace_flatten, getval

EPS, RTOL, ATOL = 1e-4, 1e-4, 1e-6

def nd(f, *args):
    unary_f = lambda x : f(*x)
    return unary_nd(unary_f, args)

def unary_nd(f, x, eps=EPS):
    vs = vspace(x)
    nd_grad = np.zeros(vs.size)
    x_flat = vs.flatten(x)
    for d in range(vs.size):
        dx = np.zeros(vs.size)
        dx[d] = eps/2
        nd_grad[d] = (   f(vs.unflatten(x_flat + dx))
                       - f(vs.unflatten(x_flat - dx))  ) / eps
    return vs.unflatten(nd_grad, True)

def indexed_function(fun, arg, index):
    def partial_function(x):
        local_arg = copy(arg)
        if isinstance(local_arg, tuple):
            local_arg = local_arg[:index] + (x,) + local_arg[index+1:]
        elif isinstance(local_arg, list):
            local_arg = local_arg[:index] + [x] + local_arg[index+1:]
        else:
            local_arg[index] = x
        return fun(local_arg)
    return partial_function

def check_equivalent(A, B, rtol=RTOL, atol=ATOL):
    A_vspace = vspace(A)
    B_vspace = vspace(B)
    A_flat = vspace_flatten(A)
    B_flat = vspace_flatten(B)
    assert A_vspace == B_vspace, \
      "VSpace mismatch:\nanalytic: {}\nnumeric: {}".format(A_vspace, B_vspace)
    assert np.allclose(vspace_flatten(A), vspace_flatten(B), rtol=rtol, atol=atol), \
        "Diffs are:\n{}.\nanalytic is:\n{}.\nnumeric is:\n{}.".format(
            A_flat - B_flat, A_flat, B_flat)

def check_grads(fun, *args):
    if not args:
        raise Exception("No args given")
    exact = tuple([grad(fun, i)(*args) for i in range(len(args))])
    args = [float(x) if isinstance(x, int) else x for x in args]
    numeric = nd(fun, *args)
    check_equivalent(exact, numeric)

def to_scalar(x):
    if isinstance(getval(x), list)  or isinstance(getval(x), tuple):
        return sum([to_scalar(item) for item in x])
    return np.sum(np.real(np.sin(x)))

def quick_grad_check(fun, arg0, extra_args=(), kwargs={}, verbose=True,
                     eps=EPS, rtol=RTOL, atol=ATOL, rs=None):
    """Checks the gradient of a function (w.r.t. to its first arg) in a random direction"""

    if verbose:
        print("Checking gradient of {0} at {1}".format(fun, arg0))

    if rs is None:
        rs = np.random.RandomState()

    random_dir = rs.standard_normal(np.shape(arg0))
    random_dir = random_dir / np.sqrt(np.sum(random_dir * random_dir))
    unary_fun = lambda x : fun(arg0 + x * random_dir, *extra_args, **kwargs)
    numeric_grad = unary_nd(unary_fun, 0.0, eps=eps)

    analytic_grad = np.sum(grad(fun)(arg0, *extra_args, **kwargs) * random_dir)

    assert np.allclose(numeric_grad, analytic_grad, rtol=rtol, atol=atol), \
        "Check failed! nd={0}, ad={1}".format(numeric_grad, analytic_grad)

    if verbose:
        print("Gradient projection OK (numeric grad: {0}, analytic grad: {1})".
              format(numeric_grad, analytic_grad))

def flatten(value):
    """Flattens any nesting of tuples, arrays, or dicts.
       Returns 1D numpy array and an unflatten function.
       Doesn't preserve mixed numeric types (e.g. floats and ints).
       Assumes dict keys are sortable."""
    vs = vspace(value)
    return vs.flatten(value), vs.unflatten

def flatten_func(func, example):
    """Flattens both the inputs to a function, and the outputs."""
    flattened_example, unflatten = flatten(example)

    def flattened_func(flattened_params, *args, **kwargs):
        output = func(unflatten(flattened_params), *args, **kwargs)
        flattened_output, _ = flatten(output)
        return flattened_output
    return flattened_func, unflatten, flattened_example
