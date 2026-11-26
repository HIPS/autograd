import numpy as onp
import pytest

import autograd.numpy as np
from autograd import grad
from autograd.test_util import check_grads


@pytest.mark.parametrize("a_shape,b_shape", [((2, 2), (2, 3)), ((4,), (2, 3)), ((), (2, 3)), ((2, 2), ())])
@pytest.mark.parametrize("complex_values", [False, True])
def test_outer_flattens_operands_in_gradients(a_shape, b_shape, complex_values):
    a = np.arange(onp.prod(a_shape), dtype=float).reshape(a_shape) + 0.5
    b = np.arange(onp.prod(b_shape), dtype=float).reshape(b_shape) - 0.3
    if complex_values:
        a = a + 0.2j
        b = b - 0.4j
    ga, gb = grad(lambda a, b: np.real(np.sum(np.outer(a, b))), (0, 1))(a, b)
    assert ga.shape == a.shape
    assert gb.shape == b.shape
    onp.testing.assert_allclose(ga, np.full(a.shape, np.sum(b)))
    onp.testing.assert_allclose(gb, np.full(b.shape, np.sum(a)))
    check_grads(np.outer, (0, 1), modes=["rev"], order=2)(a, b)
