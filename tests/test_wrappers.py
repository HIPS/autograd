import warnings
from functools import partial

import pytest

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import (
    checkpoint,
    elementwise_grad,
    grad,
    grad_and_aux,
    hessian,
    hessian_tensor_product,
    jacobian,
    make_ggnvp,
    make_hvp,
    make_jvp,
    tensor_jacobian_product,
    value_and_grad,
)
from autograd.test_util import check_equivalent, check_grads  # , nd
from autograd.tracer import isbox


def test_return_both():
    fun = lambda x: 3.0 * x**3.2
    d_fun = grad(fun)
    f_and_d_fun = value_and_grad(fun)

    test_x = 1.7
    f, d = f_and_d_fun(test_x)
    assert f == fun(test_x)
    assert d == d_fun(test_x)


def test_value_and_grad():
    rng = npr.RandomState(42)
    fun = lambda x: np.sum(np.sin(x) ** 2)
    dfun = grad(fun)
    dfun_both = value_and_grad(fun)
    x = rng.randn(5)
    assert not isbox(dfun_both(x)[0])
    check_equivalent(fun(x), dfun_both(x)[0])
    check_equivalent(dfun(x), dfun_both(x)[1])

    def fun2(x):
        return dfun_both(x)[0]

    check_grads(fun2)(x)


def test_hessian():
    rng = npr.RandomState(42)
    # Check Hessian of a quadratic function.
    D = 5
    H = rng.randn(D, D)

    def fun(x):
        return np.dot(np.dot(x, H), x)

    hess = hessian(fun)
    x = rng.randn(D)
    check_equivalent(hess(x), H + H.T)


def test_multigrad():
    def complicated_fun(a, b, c, d, e, f=1.1, g=9.0):
        return a + np.sin(b) + np.cosh(c) + np.cos(d) + np.tan(e) + f + g

    def complicated_fun_3_1(d_b):
        d, b = d_b
        return complicated_fun(A, b, C, d, E, f=F, g=G)

    A = 0.5
    B = -0.3
    C = 0.2
    D = -1.1
    E = 0.7
    F = 0.6
    G = -0.1

    wrapped = grad(complicated_fun, argnum=[3, 1])(A, B, C, D, E, f=F, g=G)
    explicit = grad(complicated_fun_3_1)((D, B))
    check_equivalent(wrapped, explicit)


def test_value_and_multigrad():
    def complicated_fun(a, b, c, d, e, f=1.1, g=9.0):
        return a + np.sin(b) + np.cosh(c) + np.cos(d) + np.tan(e) + f + g

    A = 0.5
    B = -0.3
    C = 0.2
    D = -1.1
    E = 0.7
    F = 0.6
    G = -0.1

    dfun = grad(complicated_fun, argnum=[3, 1])
    dfun_both = value_and_grad(complicated_fun, argnum=[3, 1])

    check_equivalent(complicated_fun(A, B, C, D, E, f=F, g=G), dfun_both(A, B, C, D, E, f=F, g=G)[0])

    check_equivalent(dfun(A, B, C, D, E, f=F, g=G), dfun_both(A, B, C, D, E, f=F, g=G)[1])


def test_multigrad_onearg():
    rng = npr.RandomState(42)
    fun = lambda x, y: np.sum(x + np.sin(y))
    packed_fun = lambda xy: np.sum(xy[0] + np.sin(xy[1]))
    A, B = rng.randn(3), rng.randn(3)
    check_equivalent(grad(fun, argnum=[0])(A, B), (grad(packed_fun)((A, B))[0],))


def test_elementwise_grad():
    rng = npr.RandomState(42)

    def simple_fun(a):
        return a + np.sin(a) + np.cosh(a)

    A = rng.randn(10)

    wrapped = elementwise_grad(simple_fun)(A)
    explicit = np.array([grad(simple_fun)(A[i]) for i in range(len(A))])
    check_equivalent(wrapped, explicit)


def test_elementwise_grad_multiple_args():
    rng = npr.RandomState(42)

    def simple_fun(a, b):
        return a + np.sin(a) + np.cosh(b)

    A = 0.9
    B = rng.randn(10)
    argnum = 1

    wrapped = elementwise_grad(simple_fun, argnum)(A, B)
    explicit = np.array([grad(simple_fun, argnum)(A, B[i]) for i in range(len(B))])
    check_equivalent(wrapped, explicit)


def test_hessian_tensor_product():
    rng = npr.RandomState(42)
    fun = lambda a: np.sum(np.sin(a))
    a = rng.randn(5)
    v = rng.randn(5)
    H = hessian(fun)(a)
    check_equivalent(np.dot(H, v), hessian_tensor_product(fun)(a, v))


def test_hvp():
    rng = npr.RandomState(42)
    fun = lambda a: np.sum(np.sin(a))
    a = rng.randn(5)
    v = rng.randn(5)
    H = hessian(fun)(a)
    hvp = make_hvp(fun)(a)[0]
    check_equivalent(np.dot(H, v), hvp(v))


def test_hessian_matrix_product():
    rng = npr.RandomState(42)
    fun = lambda a: np.sum(np.sin(a))
    a = rng.randn(5, 4)
    V = rng.randn(5, 4)
    H = hessian(fun)(a)
    check_equivalent(np.tensordot(H, V), hessian_tensor_product(fun)(a, V))


def test_hessian_tensor_product_3d():
    rng = npr.RandomState(42)
    fun = lambda a: np.sum(np.sin(a))
    a = rng.randn(5, 4, 3)
    V = rng.randn(5, 4, 3)
    H = hessian(fun)(a)
    check_equivalent(np.tensordot(H, V, axes=np.ndim(V)), hessian_tensor_product(fun)(a, V))


def test_tensor_jacobian_product():
    rng = npr.RandomState(42)
    # This function will have an asymmetric jacobian matrix.
    fun = lambda a: np.roll(np.sin(a), 1)
    a = rng.randn(5)
    V = rng.randn(5)
    J = jacobian(fun)(a)
    check_equivalent(np.dot(V.T, J), tensor_jacobian_product(fun)(a, V))


def test_matrix_jacobian_product():
    rng = npr.RandomState(42)
    fun = lambda a: np.roll(np.sin(a), 1)
    a = rng.randn(5, 4)
    V = rng.randn(5, 4)
    J = jacobian(fun)(a)
    check_equivalent(np.tensordot(V, J), tensor_jacobian_product(fun)(a, V))


def test_tensor_jacobian_product():
    rng = npr.RandomState(42)
    fun = lambda a: np.roll(np.sin(a), 1)
    a = rng.randn(5, 4, 3)
    V = rng.randn(5, 4)
    J = jacobian(fun)(a)
    check_equivalent(np.tensordot(V, J, axes=np.ndim(V)), tensor_jacobian_product(fun)(a, V))


def test_deprecated_defgrad_wrapper():
    rng = npr.RandomState(42)
    from autograd.core import primitive

    @primitive
    def new_mul(x, y):
        return x * y

    with warnings.catch_warnings(record=True) as w:
        new_mul.defgrad(lambda ans, x, y: lambda g: y * g)
        new_mul.defgrad(lambda ans, x, y: lambda g: x * g, argnum=1)

    def fun(x, y):
        return new_mul(x, y)

    mat1 = rng.randn(2, 2)
    mat2 = rng.randn(2, 2)
    check_grads(fun, modes=["rev"])(mat1, mat2)


def test_deprecated_defvjp_wrapper():
    rng = npr.RandomState(42)
    from autograd.core import primitive

    @primitive
    def new_mul(x, y):
        return x * y

    with warnings.catch_warnings(record=True) as w:
        new_mul.defvjp(lambda g, ans, vs, gvs, x, y: y * g)
        new_mul.defvjp(lambda g, ans, vs, gvs, x, y: x * g, argnum=1)

    def fun(x, y):
        return new_mul(x, y)

    mat1 = rng.randn(2, 2)
    mat2 = rng.randn(2, 2)
    check_grads(fun, modes=["rev"])(mat1, mat2)


def test_deprecated_defvjp_is_zero_wrapper():
    rng = npr.RandomState(42)
    from autograd.core import primitive

    @primitive
    def new_mul(x, y):
        return 0 * x * y

    with warnings.catch_warnings(record=True) as w:
        new_mul.defvjp_is_zero([0, 1])

    def fun(x, y):
        return new_mul(x, y)

    mat1 = rng.randn(2, 2)
    mat2 = rng.randn(2, 2)
    with warnings.catch_warnings(record=True) as w:
        check_grads(fun, modes=["rev"])(mat1, mat2)


def test_deprecated_quick_grad_check_wrapper():
    from autograd.util import quick_grad_check

    with warnings.catch_warnings(record=True) as w:
        quick_grad_check(lambda x, y: x**2 + y, 1.0, (2.0,))


def test_partial():
    def f(x, y):
        return x

    grad(partial(f, y=1))


def test_dtypes():
    rng = npr.RandomState(42)

    def f(x):
        return np.real(np.sum(x**2))

    # Array y with dtype np.float32
    y = rng.randn(10, 10).astype(np.float32)
    assert grad(f)(y).dtype.type is np.float32

    y = rng.randn(10, 10).astype(np.float16)
    assert grad(f)(y).dtype.type is np.float16

    y = rng.randn(10, 10).astype(np.longdouble)
    grad(f)(y)

    y = rng.randn(10, 10).astype(np.clongdouble)
    grad(f)(y)


@pytest.mark.parametrize(
    ("dtype", "condition", "x", "y"),
    [
        (int, True, 1, 2),
        (float, True, 3.0, 4.0),
        (float, False, 1, 2.0),
        (list, True, [1, 2], [3, 4]),
        (np.float32, True, np.array([1.0], dtype=np.float32), np.array([2.0], dtype=np.float32)),
        (np.float64, False, np.array([1.0], dtype=np.float64), np.array([2.0], dtype=np.float64)),
        (np.float64, False, np.array([1.0], dtype=np.float64), np.array([2.0], dtype=np.float32)),
    ],
)
def test_custom_where_dtypes(dtype, condition, x, y):
    res = np.where(condition, x, y)
    if hasattr(res, "dtype"):
        assert res.dtype.type is dtype
    else:
        assert isinstance(res, dtype)


def test_custom_where():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    condition = [True, False, True]
    expected = np.array([1.0, 5.0, 3.0])
    result = np.where(condition, x, y)
    check_equivalent(result, expected)

    # check with keyword arguments
    result_kw = np.where(condition=condition, x=x, y=y)
    check_equivalent(result_kw, expected)

    # check with mix args and kwargs
    result_mix = np.where(condition, x, y=y)
    check_equivalent(result_mix, expected)

    # check with no x and y
    result_no_xy = np.where(condition)
    assert np.all(result_no_xy == np.array([0, 2]))  # indices of True values


def checkpoint_memory():
    """This test is meant to be run manually, since it depends on
    memory_profiler and its behavior may vary."""
    try:
        from memory_profiler import memory_usage
    except ImportError:
        return

    def f(a):
        for _ in range(10):
            a = np.sin(a**2 + 1)
        return a

    checkpointed_f = checkpoint(f)

    def testfun(f, x):
        for _ in range(5):
            x = f(x)
        return np.sum(x)

    gradfun = grad(testfun, 1)

    A = npr.RandomState(0).randn(100000)
    max_usage = max(memory_usage((gradfun, (f, A))))
    max_checkpointed_usage = max(memory_usage((gradfun, (checkpointed_f, A))))

    assert max_checkpointed_usage < max_usage / 2.0


def test_make_jvp():
    rng = npr.RandomState(42)
    A = rng.randn(3, 5)
    x = rng.randn(5)
    v = rng.randn(5)
    fun = lambda x: np.tanh(np.dot(A, x))

    jvp_explicit = lambda x: lambda v: np.dot(jacobian(fun)(x), v)
    jvp = make_jvp(fun)

    check_equivalent(jvp_explicit(x)(v), jvp(x)(v)[1])


def _make_explicit_ggnvp(f, g=lambda x: 1.0 / 2 * np.dot(x, x)):
    def ggnvp_maker(x):
        J = jacobian(f)(x)
        H = hessian(g)(f(x))

        def ggnvp(v):
            return np.dot(J.T, np.dot(H, np.dot(J, v)))

        return ggnvp

    return ggnvp_maker


def test_make_ggnvp():
    rng = npr.RandomState(42)
    A = rng.randn(5, 4)
    x = rng.randn(4)
    v = rng.randn(4)

    fun = lambda x: np.dot(A, x)
    check_equivalent(make_ggnvp(fun)(x)(v), _make_explicit_ggnvp(fun)(x)(v))

    fun2 = lambda x: np.tanh(np.dot(A, x))
    check_equivalent(make_ggnvp(fun2)(x)(v), _make_explicit_ggnvp(fun2)(x)(v))


def test_make_ggnvp_nondefault_g():
    rng = npr.RandomState(42)
    A = rng.randn(5, 4)
    x = rng.randn(4)
    v = rng.randn(4)

    g = lambda y: np.sum(2.0 * y**2 + y**4)

    fun = lambda x: np.dot(A, x)
    check_equivalent(make_ggnvp(fun, g)(x)(v), _make_explicit_ggnvp(fun, g)(x)(v))

    fun2 = lambda x: np.tanh(np.dot(A, x))
    check_equivalent(make_ggnvp(fun2, g)(x)(v), _make_explicit_ggnvp(fun2, g)(x)(v))


def test_grad_and_aux():
    rng = npr.RandomState(42)
    A = rng.randn(5, 4)
    x = rng.randn(4)

    f = lambda x: (np.sum(np.dot(A, x)), x**2)
    g = lambda x: np.sum(np.dot(A, x))

    assert len(grad_and_aux(f)(x)) == 2

    check_equivalent(grad_and_aux(f)(x)[0], grad(g)(x))
    check_equivalent(grad_and_aux(f)(x)[1], x**2)


## No longer support this behavior
# def test_make_ggnvp_broadcasting():
#   A = npr.randn(4, 5)
#   x = npr.randn(10, 4)
#   v = npr.randn(10, 4)

#   fun = lambda x: np.tanh(np.dot(x, A))
#   res1 = np.stack([_make_explicit_ggnvp(fun)(xi)(vi) for xi, vi in zip(x, v)])
#   res2 = make_ggnvp(fun)(x)(v)
#   check_equivalent(res1, res2)


def test_wrapped_name_and_docs():
    def foo(x):
        pass

    assert grad.__name__ == "grad"
    # Python 3.13: Compiler now strip indents from docstrings.
    # https://docs.python.org/3.13/whatsnew/3.13.html#other-language-changes
    assert grad.__doc__.startswith(tuple(f"\n{indent}Returns a function which" for indent in ("    ", "")))
    assert grad(foo, 1).__name__ == "grad_of_foo_wrt_argnum_1"
    assert grad(foo, 1).__doc__.startswith("    grad of function foo with")
