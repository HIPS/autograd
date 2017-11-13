from __future__ import absolute_import
import warnings

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.test_util import check_grads
from autograd import grad
from numpy_utils import combo_check
npr.seed(1)

def test_dot():
    def fun( x, y): return np.dot(x, y)

    mat1 = npr.randn(10, 11)
    mat2 = npr.randn(10, 11)
    vect1 = npr.randn(10)
    vect2 = npr.randn(11)
    vect3 = npr.randn(11)

    check_grads(fun)(mat1, vect2)
    check_grads(fun)(mat1, mat2.T)
    check_grads(fun)(vect1, mat1)
    check_grads(fun)(vect2, vect3)

def test_dot_with_floats():
    def fun( x, y): return np.dot(x, y)

    mat1 = npr.randn(10, 11)
    vect1 = npr.randn(10)
    float1 = npr.randn()

    check_grads(fun)(mat1, float1)
    check_grads(fun)(float1, mat1)
    check_grads(fun)(vect1, float1)
    check_grads(fun)(float1, vect1)

# No longer supporting this
# def test_dot_method():
#     def fun(x, y): return x.dot(y)

#     mat1 = npr.randn(10, 11)
#     mat2 = npr.randn(10, 11)
#     vect1 = npr.randn(10)
#     vect2 = npr.randn(11)
#     vect3 = npr.randn(11)

#     check_grads(fun)(mat1, vect2)
#     check_grads(fun)(mat1, mat2.T)
#     check_grads(fun)(vect1, mat1)
#     check_grads(fun)(vect2, vect3)

def test_outer():
    def fun( x, y): return np.outer(x, y)

    vect2 = npr.randn(11)
    vect3 = npr.randn(11)

    check_grads(fun)(vect2, vect3)
    check_grads(fun)(vect2.T, vect3)
    check_grads(fun)(vect2.T, vect3.T)


def test_max():
    def fun(x): return np.max(x)
    mat = npr.randn(10, 11)
    check_grads(fun)(mat)

def test_max_axis():
    def fun(x): return np.max(x, axis=1)
    mat = npr.randn(3, 4, 5)
    check_grads(fun)(mat)

def test_max_axis_keepdims():
    def fun(x): return np.max(x, axis=1, keepdims=True)
    mat = npr.randn(3, 4, 5)
    check_grads(fun)(mat)

def test_min():
    def fun(x): return np.min(x)
    mat = npr.randn(10, 11)
    check_grads(fun)(mat)

def test_min_axis():
    def fun(x): return np.min(x, axis=1)
    mat = npr.randn(3, 4, 5)
    check_grads(fun)(mat)

def test_min_axis_keepdims():
    def fun(x): return np.min(x, axis=1, keepdims=True)
    mat = npr.randn(3, 4, 5)
    check_grads(fun)(mat)

def test_sum_1():
    def fun(x): return np.sum(x)
    mat = npr.randn(10, 11)
    check_grads(fun)(mat)

def test_sum_2():
    def fun(x): return np.sum(x, axis=0)
    mat = npr.randn(10, 11)
    check_grads(fun)(mat)

def test_sum_3():
    def fun(x): return np.sum(x, axis=0, keepdims=True)
    mat = npr.randn(10, 11)
    check_grads(fun)(mat)

def test_sum_with_axis_tuple():
    def fun(x): return np.sum(x, axis=(1,2))
    mat = npr.randn(10, 11, 7)
    check_grads(fun)(mat)

def test_flipud():
    def fun(x): return np.flipud(x)
    mat = npr.randn(10, 11)
    check_grads(fun)(mat)

def test_fliplr():
    def fun(x): return np.fliplr(x)
    mat = npr.randn(10, 11)
    check_grads(fun)(mat)

def test_rot90():
    def fun(x): return np.rot90(x)
    mat = npr.randn(10, 11)
    check_grads(fun)(mat)

def test_cumsum_axis0():
    def fun(x): return np.cumsum(x, axis=0)
    mat = npr.randn(10, 11)
    check_grads(fun)(mat)

def test_cumsum_axis1():
    def fun(x): return np.cumsum(x, axis=1)
    mat = npr.randn(10, 11)
    check_grads(fun)(mat)

def test_cumsum_1d():
    def fun(x): return np.cumsum(x)
    mat = npr.randn(10)
    check_grads(fun)(mat)

def test_cumsum_no_axis():
    def fun(x): return np.cumsum(x)
    mat = npr.randn(10, 11)
    check_grads(fun)(mat)

def test_non_numpy_sum():
    def fun(x, y):
        return sum([x, y])
    mat1 = npr.randn(10, 11)
    mat2 = npr.randn(10, 11)
    check_grads(fun)(mat1, mat2)

def test_mean_1():
    def fun(x): return np.mean(x)
    mat = npr.randn(10, 11)
    check_grads(fun)(mat)

def test_mean_2():
    def fun(x): return np.mean(x, axis=0)
    mat = npr.randn(10, 11)
    check_grads(fun)(mat)

def test_mean_3():
    def fun(x): return np.mean(x, axis=0, keepdims=True)
    mat = npr.randn(10, 11)
    check_grads(fun)(mat)

def test_index_ints():
    A = npr.randn(5, 6, 4)
    def fun(x): return x[3, 0, 1]
    check_grads(fun)(A)

def test_index_slice():
    A = npr.randn(5, 6, 4)
    def fun(x): return x[::-1, 2:4, :]
    check_grads(fun)(A)

def test_index_lists():
    A = npr.randn(5, 6, 4)
    def fun(x): return x[[0, 1, 2], :, :]
    check_grads(fun)(A)

def test_index_mixed():
    A = npr.randn(5, 6, 4)
    def fun(x): return x[3, 2:, [1, 3]]
    check_grads(fun)(A)

def test_vector_slice():
    A = npr.randn(5)
    def fun(x): return x[2:4]
    check_grads(fun)(A)

def test_index_slice_fanout():
    A = npr.randn(5, 6, 4)
    def fun(x):
        y = x[::-1, 2:4, :]
        z = x[::-1, 3:5, :]
        return y + z
    check_grads(fun)(A)

def test_index_multiple_slices():
    A = npr.randn(7)
    def fun(x):
        y = x[2:6]
        z = y[1:3]
        return z
    check_grads(fun)(A)

def test_reshape_method():
    A = npr.randn(5, 6, 4)
    def fun(x): return x.reshape((5 * 4, 6))
    check_grads(fun)(A)

def test_reshape_call():
    A = npr.randn(5, 6, 4)
    def fun(x): return np.reshape(x, (5 * 4, 6))
    check_grads(fun)(A)

def test_reshape_method_nolist():
    # The reshape can be called in two different ways:
    # like A.reshape((5,4)) or A.reshape(5,4).
    # This test checks that we support the second way.
    A = npr.randn(5, 6, 4)
    def fun(x): return x.reshape(5 * 4, 6)
    check_grads(fun)(A)

def test_ravel_method():
    A = npr.randn(5, 6, 4)
    def fun(x): return x.ravel()
    check_grads(fun)(A)

def test_ravel_call():
    A = npr.randn(5, 6, 4)
    def fun(x): return np.ravel(x)
    check_grads(fun)(A)

def test_flatten_method():
    A = npr.randn(5, 6, 4)
    def fun(x): return x.flatten()
    check_grads(fun)(A)

def test_simple_concatenate():
    A = npr.randn(5, 6, 4)
    B = npr.randn(4, 6, 4)
    def fun(x): return np.concatenate((A, x))
    check_grads(fun)(B)

def test_concatenate_axis_0():
    A = npr.randn(5, 6, 4)
    B = npr.randn(5, 6, 4)
    def fun(x): return np.concatenate((B, x, B))
    check_grads(fun)(A)

def test_concatenate_axis_1():
    A = npr.randn(5, 6, 4)
    B = npr.randn(5, 6, 4)
    def fun(x): return np.concatenate((B, x, B), axis=1)
    check_grads(fun)(A)

def test_concatenate_axis_1_unnamed():
    """Tests whether you can specify the axis without saying "axis=1"."""
    A = npr.randn(5, 6, 4)
    B = npr.randn(5, 6, 4)
    def fun(x): return np.concatenate((B, x, B), 1)
    check_grads(fun)(A)

def test_trace():
    def fun(x): return np.trace(x, offset=offset)
    mat = npr.randn(10, 11)
    offset = npr.randint(-9,11)
    check_grads(fun)(mat)

def test_trace2():
    def fun(x): return np.trace(x, offset=offset)
    mat = npr.randn(11, 10)
    offset = npr.randint(-9,11)
    check_grads(fun)(mat)

def test_trace_extradims():
    def fun(x): return np.trace(x, offset=offset)
    mat = npr.randn(5,6,4,3)
    offset = npr.randint(-5,6)
    check_grads(fun)(mat)

# TODO: Allow axis1, axis2 args.
# def test_trace_extradims2():
#     def fun(x): return np.trace(x, offset=offset, axis1=3,axis2=2)
#     mat = npr.randn(5,6,4,3)
#     offset = npr.randint(-5,6)
#     check_grads(fun)(mat)

def test_diag():
    def fun(x): return np.diag(x)
    mat = npr.randn(10, 10)
    check_grads(fun)(mat)

def test_transpose():
    def fun(x): return x.T
    mat = npr.randn(8, 8)
    check_grads(fun)(mat)

def test_roll():
    def fun(x): return np.roll(x, 2, axis=1)
    mat = npr.randn(4, 5)
    check_grads(fun)(mat)

def test_roll_no_axis():
    def fun(x): return np.roll(x, 2, axis=1)
    mat = npr.randn(4, 5)
    check_grads(fun)(mat)

def test_triu():
    def fun(x): return np.triu(x, k = 2)
    mat = npr.randn(5, 5)
    check_grads(fun)(mat)

def test_tril():
    def fun(x): return np.tril(x, k = 2)
    mat = npr.randn(5, 5)
    check_grads(fun)(mat)

def test_clip():
    def fun(x): return np.clip(x, a_min=0.1, a_max=1.1)
    mat = npr.randn(5, 5)
    check_grads(fun)(mat)

def test_prod_1():
    def fun(x): return np.prod(x)
    mat = npr.randn(2, 3)**2 / 10.0 + 0.1  # Gradient unstable when zeros are present.
    check_grads(fun)(mat)

def test_prod_2():
    def fun(x): return np.prod(x, axis=0)
    mat = npr.randn(2, 3)**2 + 0.1
    check_grads(fun)(mat)

def test_prod_3():
    def fun(x): return np.prod(x, axis=0, keepdims=True)
    mat = npr.randn(2, 3)**2 + 0.1
    check_grads(fun)(mat)

def test_prod_4():
    def fun(x): return np.prod(x)
    mat = npr.randn(7)**2 + 0.1
    check_grads(fun)(mat)

def test_1d_array():
    def fun(x):
        return np.array([x, x * 1.0, x + 2.5])
    check_grads(fun)(3.0)

def test_2d_array():
    def fun(x):
        return np.array([[x   , x * 1.0, x + 2.5],
                                   [x**2, x      , x / 2.0]])

    check_grads(fun)(3.0)

def test_1d_array_fanout():
    def fun(x):
        A = np.array([x, x * 1.0, x + 2.5])
        return A + A
    check_grads(fun)(3.0)

def test_2d_array_fanout():
    def fun(x):
        A = np.array([[x   , x * 1.0, x + 2.5],
                      [x**2, x      , x / 2.0]])
        return A + A

    check_grads(fun)(3.0)

def test_array_from_scalar():
    def fun(x):
        return np.array(x)

    check_grads(fun)(3.0)

def test_array_from_arrays():
    def fun(x):
        return np.array([x, x])

    A = npr.randn(3, 2)
    check_grads(fun)(A)

def test_array_from_arrays_2():
    def fun(x):
        return np.array([[2*x, x + 1],
                                   [x  ,     x]])
    A = npr.randn(3, 2)
    check_grads(fun)(A)

def test_len():
    def fun(x):
        assert len(x) == 3
        return x
    A = npr.randn(3, 2)
    check_grads(fun)(A)

def test_r_basic():
    with warnings.catch_warnings(record=True) as w:
        def fun(x):
            c = npr.randn(3, 2)
            b = np.r_[x]
            return b
        A = npr.randn(3, 2)
        check_grads(fun)(A)

def test_r_double():
    with warnings.catch_warnings(record=True) as w:
        def fun(x):
            c = npr.randn(3, 2)
            b = np.r_[x, x]
            return b
        A = npr.randn(3, 2)
        check_grads(fun)(A)

def test_no_relation():
    with warnings.catch_warnings(record=True) as w:
        c = npr.randn(3, 2)
        def fun(x):
            return c
        A = npr.randn(3, 2)
        check_grads(fun)(A)

def test_r_no_relation():
    with warnings.catch_warnings(record=True) as w:
        c = npr.randn(3, 2)
        def fun(x):
            b = np.r_[c]
            return b
        A = npr.randn(3, 2)
        check_grads(fun)(A)

def test_r_node_and_const():
    with warnings.catch_warnings(record=True) as w:
        c = npr.randn(3, 2)
        def fun(x):
            b = np.r_[x, c]
            return b
        A = npr.randn(3, 2)
        check_grads(fun)(A)

def test_r_mixed():
    with warnings.catch_warnings(record=True) as w:
        c = npr.randn(3, 2)
        def fun(x):
            b = np.r_[x, c, x]
            return b
        A = npr.randn(3, 2)
        check_grads(fun)(A)

def test_r_slicing():
    with warnings.catch_warnings(record=True) as w:
        c = npr.randn(10)
        def fun(x):
            b = np.r_[x, c, 1:10]
            return b
        A = npr.randn(10)
        check_grads(fun)(A)

def test_c_():
    with warnings.catch_warnings(record=True) as w:
        c = npr.randn(3, 2)
        def fun(x):
            b = np.c_[x, c, x]
            return b
        A = npr.randn(3, 2)
        check_grads(fun)(A)

def test_c_mixed():
    with warnings.catch_warnings(record=True) as w:
        c = npr.randn(3, 2)
        def fun(x):
            b = np.c_[x, c, x]
            return b
        A = npr.randn(3, 2)
        check_grads(fun)(A)

def test_var_ddof():
    B = npr.randn(3)
    C = npr.randn(3, 4)
    D = npr.randn(1, 3)
    combo_check(np.var, (0,))([B, C, D], axis=[None], keepdims=[True, False], ddof=[0, 1])
    combo_check(np.var, (0,))([C, D], axis=[None, 1], keepdims=[True, False], ddof=[2])

def test_std_ddof():
    B = npr.randn(3)
    C = npr.randn(3, 4)
    D = npr.randn(1, 3)
    combo_check(np.std, (0,))([B, C, D], axis=[None], keepdims=[True, False], ddof=[0, 1])
    combo_check(np.std, (0,))([C, D], axis=[None, 1], keepdims=[True, False], ddof=[2])

def test_where():
    def fun(x, y):
        b = np.where(C, x, y)
        return b
    C = npr.randn(4, 5) > 0
    A = npr.randn(4, 5)
    B = npr.randn(4, 5)
    check_grads(fun)(A, B)

def test_squeeze_func():
    A = npr.randn(5, 1, 4)
    def fun(x): return np.squeeze(x)
    check_grads(fun)(A)

def test_squeeze_method():
    A = npr.randn(5, 1, 4)
    def fun(x): return x.squeeze()
    check_grads(fun)(A)

def test_repeat():
    A = npr.randn(5, 3, 4)
    def fun(x): return np.repeat(x, 2, axis=1)
    check_grads(fun)(A)

def test_repeat_axis1_rep1():
    A = npr.randn(5, 3, 4)
    def fun(x): return np.repeat(x, 1, axis=1)
    check_grads(fun)(A)

def test_repeat_axis0():
    A = npr.randn(5, 3)
    def fun(x): return np.repeat(x, 2, axis=0)
    check_grads(fun)(A)

def test_repeat_1d_axis0():
    A = npr.randn(5)
    def fun(x): return np.repeat(x, 2, axis=0)
    check_grads(fun)(A)

def test_repeat_axis0_rep1():
    A = npr.randn(5, 1)
    def fun(x): return np.repeat(x, 1, axis=0)
    check_grads(fun)(A)

def test_expand_dims():
    A = npr.randn(5, 1, 4)
    def fun(x): return np.expand_dims(x, 2)
    check_grads(fun)(A)

def test_tensordot_kwargs_by_position():
    def fun(x):
        return np.tensordot(x * np.ones((2,2)),
                            x * np.ones((2,2)), 2)
    grad(fun)(1.0)

def test_multi_index():
    A = npr.randn(3)
    fun = lambda x: np.sum(x[[0, 0]])
    check_grads(fun)(A)

def test_multi_index2():
    A = npr.randn(3)
    fun = lambda x: np.sum(x[[0, 1, 0]])
    check_grads(fun)(A)

def test_index_dot_slices():
    A = npr.randn(4)
    def fun(x): return np.dot(x[:2], x[2:])
    check_grads(fun)(A)

#def test_index_exp_slicing():
#    def fun(x):
#        b = np.index_exp[x, x]
#        return b
#    A = npr.randn(10, 1)
#    check_grads(fun)(A)

#def test_s_slicing():
#    def fun(x):
#        b = np.s_[x, x]
#        return b
#    A = npr.randn(10, 1)
#    check_grads(fun)(A)

# TODO:
# getitem

def test_cast_to_int():
    inds = np.ones(5)[:,None]

    def fun(W):
        # glue W and inds together
        glued_together = np.concatenate((W, inds), axis=1)

        # separate W and inds back out
        new_W = W[:,:-1]
        new_inds = np.int64(W[:,-1])

        assert new_inds.dtype == np.int64
        return new_W[new_inds].sum()

    W = np.random.randn(5, 10)
    check_grads(fun)(W)

def test_make_diagonal():
    def fun(D):
        return np.make_diagonal(D, axis1=-1, axis2=-2)

    D = np.random.randn(4)
    A = np.make_diagonal(D, axis1=-1, axis2=-2)
    assert np.allclose(np.diag(A), D)
    check_grads(fun)(D)

    D = np.random.randn(3, 4)
    A = np.make_diagonal(D, axis1=-1, axis2=-2)
    assert all([np.allclose(np.diag(A[i]), D[i]) for i in range(3)])
    check_grads(fun)(D)

def test_diagonal():
    def fun(D):
        return np.diagonal(D, axis1=-1, axis2=-2)

    D = np.random.randn(4, 4)
    A = np.make_diagonal(D, axis1=-1, axis2=-2)
    check_grads(fun)(D)

    D = np.random.randn(3, 4, 4)
    A = np.make_diagonal(D, axis1=-1, axis2=-2)
    check_grads(fun)(D)

def test_nan_to_num():
    y = np.array([0., np.nan, np.inf, -np.inf])
    fun = lambda x: np.sum(np.sin(np.nan_to_num(x + y)))

    x = np.random.randn(4)
    check_grads(fun)(x)

# TODO(mattjj): np.frexp returns a pair of ndarrays and the second is an int
# type, for which there is currently no vspace registered
#def test_frexp():
#    fun = lambda x: np.frexp(x)[0]
#    A = 1.2 #np.random.rand(4,3) * 0.8 + 2.1
#    check_grads(fun)(A)

def test_max_equal_values():
    def fun(x): return np.max(np.array([x, x]))
    check_grads(fun)(1.0)

def test_max_equal_values_2d():
    def fun(x): return np.max(np.array([[x, x  ],
                                                  [x, 0.5]]), axis=1)
    check_grads(fun)(1.0)
    check_grads(fun)(-1.0)

def test_min_3_way_equality():
    def fun(x): return np.min(np.array([[x,     x,   x],
                                                  [x,   0.5, 0.5],
                                                  [0.5, 0.5, 0.5],
                                                  [x,     x, 0.5]]), axis=0)
    check_grads(fun)(1.0)
    check_grads(fun)(-1.0)

def test_maximum_equal_values():
    def fun(x): return np.maximum(x, x)
    check_grads(fun)(1.0)

def test_maximum_equal_values_2d():
    def fun(x): return np.maximum(
            np.array( [x,   x, 0.5]),
            np.array([[x, 0.5,   x],
                      [x,   x, 0.5]]))

    check_grads(fun)(1.0)
    check_grads(fun)(-1.0)
    check_grads(fun)(2.0)

def test_linspace():
    for num in [0, 1, 5]:
        def fun(x, y): return np.linspace(x, y, num)

        check_grads(fun)(1.2, 3.4)
        check_grads(fun)(1.2, -3.4)
        check_grads(fun)(1.2, 1.2)

def test_astype():
    x = np.arange(3, dtype='float32')
    def f(x): return np.sum(np.sin(x.astype('float64')))
    assert grad(f)(x).dtype == np.dtype('float32')
