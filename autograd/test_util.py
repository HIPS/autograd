from functools import partial
from itertools import product
from .core import make_vjp, make_jvp, vspace
from .util import subvals
from .wrap_util import unary_to_nary, get_name

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

def check_vjp(f, x):
    vjp, y = make_vjp(f, x)
    jvp = make_numerical_jvp(f, x)
    x_vs, y_vs = vspace(x), vspace(y)
    x_v, y_v = x_vs.randn(), y_vs.randn()

    vjp_y = x_vs.covector(vjp(y_vs.covector(y_v)))
    assert vspace(vjp_y) == x_vs
    vjv_exact   = x_vs.inner_prod(x_v, vjp_y)
    vjv_numeric = y_vs.inner_prod(y_v, jvp(x_v))
    assert scalar_close(vjv_numeric, vjv_exact), \
        ("Derivative (VJP) check of {} failed with arg {}:\n"
         "analytic: {}\nnumeric:  {}".format(
            get_name(f), x, vjv_exact, vjv_numeric))

def check_jvp(f, x):
    jvp = make_jvp(f, x)
    jvp_numeric = make_numerical_jvp(f, x)
    x_v = vspace(x).randn()
    check_equivalent(jvp(x_v)[1], jvp_numeric(x_v))

def check_equivalent(x, y):
    x_vs, y_vs = vspace(x), vspace(y)
    assert x_vs == y_vs, "VSpace mismatch:\nx: {}\ny: {}".format(x_vs, y_vs)
    v = x_vs.randn()
    assert scalar_close(x_vs.inner_prod(x, v), x_vs.inner_prod(y, v)), \
        "Value mismatch:\nx: {}\ny: {}".format(x, y)

@unary_to_nary
def check_grads(f, x, modes=['fwd', 'rev'], order=2):
    assert all(m in ['fwd', 'rev'] for m in modes)
    if 'fwd' in modes:
        check_jvp(f, x)
        if order > 1:
            grad_f = lambda x, v: make_jvp(f, x)(v)[1]
            grad_f.__name__ = 'jvp_{}'.format(get_name(f))
            v = vspace(x).randn()
            check_grads(grad_f, (0, 1), modes, order=order-1)(x, v)
    if 'rev' in modes:
        check_vjp(f, x)
        if order > 1:
            grad_f = lambda x, v: make_vjp(f, x)[0](v)
            grad_f.__name__ = 'vjp_{}'.format(get_name(f))
            v = vspace(f(x)).randn()
            check_grads(grad_f, (0, 1), modes, order=order-1)(x, v)

def combo_check(fun, *args, **kwargs):
    # Tests all combinations of args and kwargs given.
    _check_grads = lambda f: check_grads(f, *args, **kwargs)
    def _combo_check(*args, **kwargs):
        kwarg_key_vals = [[(k, x) for x in xs] for k, xs in kwargs.items()]
        for _args in product(*args):
            for _kwargs in product(*kwarg_key_vals):
                _check_grads(fun)(*_args, **dict(_kwargs))
    return _combo_check
