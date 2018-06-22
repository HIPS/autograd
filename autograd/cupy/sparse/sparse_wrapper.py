import cupy.sparse as _sp
from autograd.extend import primitive
import cupy as _cp

# ----- definition for coo_matrix ----- #
def coo_matrix(arg1, *args, **kwargs):
    return sparse_matrix_from_args('coo', arg1,  *args, **kwargs)


# ----- definition for csr_matrix ----- #
def csr_matrix(arg1, *args, **kwargs):
    return sparse_matrix_from_args('csr', arg1, *args, **kwargs)

# ----- definition for csc_matrix ----- #
def csc_matrix(arg1, *args, **kwargs):
    return sparse_matrix_from_args('csc', arg1, *args, **kwargs)

# ----- definition for dia_matrix ----- #
def dia_matrix(arg1, *args, **kwargs):
    return sparse_matrix_from_args('dia', arg1, *args, **kwargs)



@primitive
def sparse_matrix_from_args(type, arg1, *args, **kwargs):
    if type == 'coo':
        return _sp.coo_matrix(arg1, *args, **kwargs)
    elif type == 'csr':
        return _sp.csr_matrix(arg1, *args, **kwargs)
    elif type == 'csc':
        return _sp.csc_matrix(arg1, *args, **kwargs)
    elif type == 'dia':
        return _sp.dia_matrix(arg1, *args, **kwargs)


@primitive
def dot(sparse, dense):
    assert not isinstance(sparse, _cp.ndarray), 'the sparse array must be the first argument'
    return sparse.dot(dense)


@primitive
def eye(N):
    return _cp.sparse.eye(N)
