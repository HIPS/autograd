import autograd.cupy as cp
from autograd import elementwise_grad as egrad
import pytest
from autograd import grad
import pdb
import numpy as np


# ----- pytest fixture for sparse arrays ----- #
@pytest.fixture
def sparse():
    data = cp.array([1, 2, 3]).astype('float32')
    rows = cp.array([1, 2, 3]).astype('float32')
    cols = cp.array([1, 3, 4]).astype('float32')

    return cp.sparse.coo_matrix((data, (rows, cols)))
    


# ----- tests for array creation ----- #
@pytest.mark.works
@pytest.mark.cupy_sparse
def test_sparse_coo_matrix():
    """This just has to not error out."""
    data = cp.array([1, 2, 3]).astype('float32')
    rows = cp.array([1, 2, 3]).astype('float32')
    cols = cp.array([1, 3, 4]).astype('float32')
    a = cp.sparse.coo_matrix((data, (rows, cols)))


# ----- tests for array multiplication ----- #
@pytest.mark.test
@pytest.mark.cupy_sparse
def test_sparse_dense_multiplication(sparse):
    """This just has to not error out."""
    dense = cp.random.random(size=(3, 4))
    cp.dot(sparse, dense)
