import autograd.numpy as np
import itertools as it
from autograd import grad
from copy import copy

def nd(f, *args):
    unary_f = lambda x : f(*x)
    return unary_nd(unary_f, args)

def unary_nd(f, x, eps=1e-4):
    if isinstance(x, np.ndarray):
        nd_grad = np.zeros(x.shape)
        for dims in it.product(*map(range, x.shape)):
            nd_grad[dims] = unary_nd(indexed_function(f, x, dims), x[dims])
        return nd_grad
    elif isinstance(x, tuple):
        return tuple([unary_nd(indexed_function(f, tuple(x), i), x[i])
                      for i in range(len(x))])
    elif isinstance(x, dict):
        return {k : unary_nd(indexed_function(f, x, k), v) for k, v in x.iteritems()}
    elif isinstance(x, list):
        return [unary_nd(indexed_function(f, x, i), v) for i, v in enumerate(x)]
    else:
        return (f(x + eps/2) - f(x - eps/2)) / eps

def indexed_function(fun, arg, index):
    def partial_function(x):
        local_arg = copy(arg)
        if isinstance(local_arg, tuple):
            local_arg = local_arg[:index] + (x,) + local_arg[index+1:]
        else:
            local_arg[index] = x
        return fun(local_arg)
    return partial_function

def eq_class(dtype):
    return float if dtype == np.float64 else dtype

def check_equivalent(A, B, rtol=1e-4, atol=1e-6):
    assert eq_class(type(A)) == eq_class(type(B)),\
        "Types are: {0} and {1}".format(eq_class(type(A)), eq_class(type(B)))
    if isinstance(A, (tuple, list)):
        for a, b in zip(A, B): check_equivalent(a, b)
    elif isinstance(A, dict):
        assert len(A) == len(B)
        for k in A: check_equivalent(A[k], B[k])
    else:
        if isinstance(A, np.ndarray):
            assert A.shape == B.shape, "Shapes are {0} and {1}".format(A.shape, B.shape)
        assert np.allclose(A, B, rtol=rtol, atol=atol), "Diffs are: {0}".format(A - B)

def check_grads(fun, *args):
    if not args:
        raise Exception("No args given")
    B = tuple([grad(fun, i)(*args) for i in range(len(args))])
    A = nd(fun, *args)

    check_equivalent(A, B)

def to_scalar(x):
    return np.sum(np.sin(x))

def quick_grad_check(fun, arg0, extra_args=(), kwargs={}, verbose=True,
                     eps=1e-4, rtol=1e-4, atol=1e-6, rs=None):
    """Checks the gradient of a function (w.r.t. to its first arg) in a random direction"""

    if verbose:
        print "Checking gradient of {0} at {1}".format(fun, arg0)

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
        print "Gradient projection OK (numeric grad: {0}, analytic grad: {1})".format(
            numeric_grad, analytic_grad)
