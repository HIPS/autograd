import autograd.cupy as cp
from autograd import elementwise_grad as egrad
import pytest
from autograd import grad
import pdb
import numpy as np
from autograd.test_util import check_grads


# ----- pytest fixture for sparse arrays ----- #
@pytest.fixture
def eye():
    return cp.sparse.eye(5).tocsr()


# ----- tests for array creation ----- #
@pytest.mark.works
@pytest.mark.cupy_sparse
def test_sparse_coo_matrix(sparse):
    """This just has to not error out."""
    data = cp.array([1, 2, 3]).astype('float32')
    rows = cp.array([1, 2, 3]).astype('float32')
    cols = cp.array([1, 3, 4]).astype('float32')
    print(sparse.shape)


# ----- tests for array multiplication ----- #
@pytest.mark.cupy_sparse
def test_sparse_dense_multiplication(sparse):
    """This just has to not error out."""
    dense = cp.random.random(size=(5, 4))
    sparse.dot(dense)


@pytest.mark.test
@pytest.mark.cupy_sparse
def test_sparse_dot(eye):
    dense = cp.random.random(size=(5, 5))

    def fun(x):
        return cp.sparse.dot(x, dense)

    check_grads(fun)(eye)
