import string

import numpy as onp
import pytest

import autograd.numpy as np
from autograd import grad, hessian
from autograd.test_util import check_grads


def test_trace_gradient():
    x = onp.arange(9.0).reshape(3, 3)
    onp.testing.assert_allclose(grad(lambda a: np.einsum("ii->", a))(x), onp.eye(3))


@pytest.mark.parametrize(
    "sub,shape",
    [
        ("ii->", (3, 3)),
        ("ii", (3, 3)),
        ("ii->i", (3, 3)),
        ("iii->", (2, 2, 2)),
        ("iiii->i", (2, 2, 2, 2)),
        ("iijj->", (2, 2, 3, 3)),
        ("ijikj->k", (2, 3, 2, 2, 3)),
        ("...ii->...", (2, 3, 3)),
        ("ii...->...", (3, 3, 2)),
        ("i...i->...", (3, 2, 3)),
        ("...ii->...", (3, 3)),
        ("ii->", (0, 0)),
        ("...ii->...", (0, 3, 3)),
        ("ii->", (1, 1)),
    ],
)
@pytest.mark.parametrize("complex_", [False, True])
def test_repeated_derivatives(sub, shape, complex_):
    x = onp.random.randn(*shape)
    if complex_:
        x = x + 1j * onp.random.randn(*shape)
    check_grads(lambda a: np.einsum(sub, a), order=2)(x)


@pytest.mark.parametrize("shape_a,shape_b", [((1, 1), (3,)), ((3, 3), (1,)), ((3, 3), (3,))])
def test_shared_label_broadcast(shape_a, shape_b):
    x = onp.random.randn(*shape_a)
    y = onp.random.randn(*shape_b)
    check_grads(lambda a, b: np.einsum("ii,i->", a, b), argnum=(0, 1), order=2)(x, y)


@pytest.mark.parametrize(
    "labels,shape", [([0, 0], (3, 3)), ([0, Ellipsis, 0], (3, 2, 3)), ([Ellipsis, 0, 0], (2, 3, 3))]
)
def test_list_syntax(labels, shape):
    output = [Ellipsis] if Ellipsis in labels else []
    check_grads(lambda a: np.einsum(a, labels, output), order=2)(onp.random.randn(*shape))


def test_nonlinear_second_order():
    check_grads(lambda a: np.sum(np.einsum("ii->i", a) ** 3), order=2)(onp.random.randn(3, 3))


def test_repeated_other_operand():
    a = onp.random.randn(3, 3)
    b = onp.random.randn(3, 3)
    check_grads(lambda x, y: np.einsum("ii,ii->", x, y), argnum=(0, 1), order=2)(a, b)


@pytest.mark.parametrize("list_syntax", [False, True])
def test_third_repeated_operand(list_syntax):
    a = onp.random.randn(2, 2, 2)
    b = onp.random.randn(3, 2, 2)
    c = onp.random.randn(3, 3, 2)
    if list_syntax:
        f = lambda x, y, z: np.einsum(x, [0, 1, 2], y, [3, 1, 0], z, [3, 3, 0], [3, 2, 0])
    else:
        f = lambda x, y, z: np.einsum("ijk,lji,lli->lki", x, y, z)
    check_grads(f, argnum=(0, 1, 2), order=2)(a, b, c)


@pytest.mark.parametrize("optimize", [False, True, "greedy", "optimal"])
def test_optimization(optimize):
    x = onp.random.randn(3, 3)
    y = onp.random.randn(3)
    check_grads(lambda a, b: np.einsum("ii,i->", a, b, optimize=optimize), argnum=(0, 1), order=2)(x, y)


def test_hessian_exact():
    x = onp.arange(9.0).reshape(3, 3)
    expected = onp.zeros((3, 3, 3, 3))
    for i in range(3):
        expected[i, i, i, i] = 6 * x[i, i]
    onp.testing.assert_allclose(hessian(lambda a: np.sum(np.einsum("ii->i", a) ** 3))(x), expected)


@pytest.mark.parametrize("dtype", [onp.float32, onp.float64, onp.complex64, onp.complex128])
def test_gradient_dtype(dtype):
    x = onp.ones((3, 3), dtype=dtype)
    gradient = grad(lambda a: np.real(np.einsum("ii->", a)))(x)
    assert gradient.dtype == x.dtype
    onp.testing.assert_allclose(gradient, onp.eye(3))


@pytest.mark.parametrize(
    "shapes", [((3, 3), (2, 3)), ((1, 3, 3), (2, 3)), ((2, 3, 3), (1, 3)), ((0, 3, 3), (1, 3))]
)
def test_repeated_batch_broadcast(shapes):
    a = onp.random.randn(*shapes[0])
    b = onp.random.randn(*shapes[1])
    check_grads(lambda x, y: np.einsum("...ii,...i->...", x, y), argnum=(0, 1), order=2)(a, b)


def test_invalid_repeated_dimensions():
    with pytest.raises(ValueError):
        grad(lambda a: np.einsum("ii->", a))(onp.ones((2, 3)))


@pytest.mark.skipif(
    onp.lib.NumpyVersion(onp.__version__) < "2.0.0",
    reason="NumPy 1.x einsum supports at most 32 iterator axes",
)
def test_no_fresh_labels_needed():
    labels = string.ascii_letters[1:]
    sub = "aa," + labels[:26] + "," + labels[26:] + "->"
    # Split the labels to avoid relying on a single very high-rank array.
    other_a = onp.ones((1,) * 26)
    other_b = onp.ones((1,) * 25)
    onp.testing.assert_allclose(
        grad(lambda a: np.einsum(sub, a, other_a, other_b))(onp.ones((2, 2))), onp.eye(2)
    )


@pytest.mark.parametrize("weight", [onp.inf, -onp.inf, onp.nan])
def test_nonfinite_gradient_stays_diagonal(weight):
    x = onp.ones((3, 3))
    weights = onp.full(3, weight)
    result = grad(lambda a: np.einsum("ii,i->", a, weights))(x)
    onp.testing.assert_array_equal(result[~onp.eye(3, dtype=bool)], 0)
    onp.testing.assert_array_equal(onp.diag(result), weights)


@pytest.mark.parametrize("first_complex", [False, True])
def test_mixed_real_complex(first_complex):
    a = onp.random.randn(3, 3)
    b = onp.random.randn(3)
    if first_complex:
        a = a + 1j * onp.random.randn(3, 3)
    else:
        b = b + 1j * onp.random.randn(3)
    check_grads(lambda x, y: np.einsum("ii,i->", x, y), argnum=(0, 1), order=2)(a, b)


@pytest.mark.parametrize(
    "sub,shape", [("...ii->...", (2, 1, 3, 3)), ("i...i->...", (3, 2, 1, 3)), ("ii...->...", (3, 3, 2, 1))]
)
def test_multiple_ellipsis_axes(sub, shape):
    check_grads(lambda a: np.einsum(sub, a), order=2)(onp.random.randn(*shape))
