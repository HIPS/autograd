from __future__ import absolute_import
from __future__ import print_function
import autograd.numpy as np
import itertools as it
from autograd.convenience_wrappers import grad, safe_type
from autograd.core import vspace, flatten
from copy import copy
from autograd.container_types import ListNode, TupleNode, make_tuple
from builtins import map, range, zip
from future.utils import iteritems

array_types = (np.ndarray,)
EPS, RTOL, ATOL = 1e-4, 1e-4, 1e-6

def nd(f, *args):
    unary_f = lambda x : f(*x)
    return unary_nd(unary_f, args)

def unary_nd(f, x, eps=EPS):
    if isinstance(x, array_types):
        if np.iscomplexobj(x):
            nd_grad = np.zeros(x.shape) + 0j
        else:
            nd_grad = np.zeros(x.shape)
        for dims in it.product(*list(map(range, x.shape))):
            nd_grad[dims] = unary_nd(indexed_function(f, x, dims), x[dims])
        return nd_grad
    elif isinstance(x, tuple):
        return tuple([unary_nd(indexed_function(f, tuple(x), i), x[i])
                      for i in range(len(x))])
    elif isinstance(x, dict):
        return {k : unary_nd(indexed_function(f, x, k), v) for k, v in iteritems(x)}
    elif isinstance(x, list):
        return [unary_nd(indexed_function(f, x, i), v) for i, v in enumerate(x)]
    elif np.iscomplexobj(x):
        result = (f(x +    eps/2) - f(x -    eps/2)) / eps \
            - 1j*(f(x + 1j*eps/2) - f(x - 1j*eps/2)) / eps
        return type(safe_type(x))(result)
    else:
        return type(safe_type(x))((f(x + eps/2) - f(x - eps/2)) / eps)

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
    A_flat = flatten(A)
    B_flat = flatten(B)
    assert A_vspace == B_vspace, \
      "VSpace mismatch:\nanalytic: {}\nnumeric: {}".format(A_vspace, B_vspace)
    assert np.allclose(flatten(A), flatten(B), rtol=rtol, atol=atol), \
        "Diffs are:\n{}.\nanalytic is:\n{}.\nnumeric is:\n{}.".format(
            A_flat - B_flat, A_flat, B_flat)

def check_grads(fun, *args):
    if not args:
        raise Exception("No args given")
    exact = tuple([grad(fun, i)(*args) for i in range(len(args))])
    numeric = nd(fun, *args)
    check_equivalent(exact, numeric)

def to_scalar(x):
    if isinstance(x, list)  or isinstance(x, ListNode) or \
       isinstance(x, tuple) or isinstance(x, TupleNode):
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
