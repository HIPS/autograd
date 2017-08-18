import itertools as it
from .vspace import vspace
from .core import make_vjp
from .util import subvals

TOL  = 1e-6
RTOL = 1e-6
def scalar_close(a, b):
    return abs(a - b) < TOL or  abs(a - b) / abs(a + b) < RTOL

EPS  = 1e-6
def make_numerical_jvp(f, x):
    y = f(x)
    x_vs, y_vs = vspace(x), vspace(y)
    def jvp(v):
        # (f(x + v*eps/2) - f(x - v*eps/2)) / eps
        f_x_plus  = f(x_vs.add(x, x_vs.scalar_mul(v,  EPS/2)))
        f_x_minus = f(x_vs.add(x, x_vs.scalar_mul(v, -EPS/2)))
        neg_f_x_minus = y_vs.scalar_mul(f_x_minus, -1.0)
        return y_vs.scalar_mul(y_vs.add(f_x_plus, neg_f_x_minus), 1.0 / EPS)
    return jvp

def check_vjp_unary(f, x):
    vjp, y = make_vjp(f, x)
    jvp = make_numerical_jvp(f, x)
    x_vs, y_vs = vspace(x), vspace(y)
    x_v, y_v = x_vs.randn(), y_vs.randn()

    vjp_y = x_vs.covector(vjp(y_vs.covector(y_v)))
    vjv_numeric = x_vs.inner_prod(x_v, vjp_y)
    vjv_exact   = y_vs.inner_prod(y_v, jvp(x_v))
    assert scalar_close(vjv_numeric, vjv_exact), \
        "Derivative check failed with arg {}:\nanalytic: {}\nnumeric: {}".format(
            x, vjv_numeric, vjv_exact)

def check_vjp(f, argnums=None, order=2):
    def _check_vjp(*args, **kwargs):
        if not order: return
        _argnums = argnums if argnums else range(len(args))
        x = tuple(args[argnum] for argnum in _argnums)
        f_unary = lambda x: f(*subvals(args, zip(_argnums, x)), **kwargs)
        check_vjp_unary(f_unary, x)

        v = vspace(f_unary(x)).randn()
        f_unary_vjp = lambda x, v: make_vjp(f_unary, x)[0](v)
        check_vjp(f_unary_vjp, order=order-1)(x, v)
    return _check_vjp

# backwards compatibility
def check_grads(f, *args): return check_vjp(f, order=1)(*args)
def nd(f, *args):
    return [make_numerical_jvp(lambda args: f(*args), args)(v)
            for v in vspace(args).standard_basis()]

def check_equivalent(x, y):
    x_vs, y_vs = vspace(x), vspace(y)
    assert x_vs == y_vs, "VSpace mismatch:\nx: {}\ny: {}".format(x_vs, y_vs)
    v = x_vs.randn()
    assert scalar_close(x_vs.inner_prod(x, v), x_vs.inner_prod(y, v)), \
        "Value mismatch:\nx: {}\ny: {}".format(x, y)

def combo_check(fun, argnums, *args, **kwargs):
    # Tests all combinations of args given.
    args = list(args)
    kwarg_key_vals = [[(key, val) for val in kwargs[key]] for key in kwargs]
    num_args = len(args)
    for args_and_kwargs in it.product(*(args + kwarg_key_vals)):
        cur_args = args_and_kwargs[:num_args]
        cur_kwargs = dict(args_and_kwargs[num_args:])
        check_vjp(fun, argnums)(*cur_args, **cur_kwargs)
