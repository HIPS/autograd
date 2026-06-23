import warnings

from numpy_utils import combo_check

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.test_util import check_grads


def test_numpy_version():
    import numpy

    assert np.__version__ == numpy.__version__


def test_dot():
    rng = npr.RandomState(42)

    def fun(x, y):
        return np.dot(x, y)

    mat1 = rng.randn(10, 11)
    mat2 = rng.randn(10, 11)
    vect1 = rng.randn(10)
    vect2 = rng.randn(11)
    vect3 = rng.randn(11)

    check_grads(fun)(mat1, vect2)
    check_grads(fun)(mat1, mat2.T)
    check_grads(fun)(vect1, mat1)
    check_grads(fun)(vect2, vect3)


def test_dot_with_floats():
    rng = npr.RandomState(42)

    def fun(x, y):
        return np.dot(x, y)

    mat1 = rng.randn(10, 11)
    vect1 = rng.randn(10)
    float1 = rng.randn()

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
    rng = npr.RandomState(42)

    def fun(x, y):
        return np.outer(x, y)

    vect2 = rng.randn(11)
    vect3 = rng.randn(11)

    check_grads(fun)(vect2, vect3)
    check_grads(fun)(vect2.T, vect3)
    check_grads(fun)(vect2.T, vect3.T)


def test_max():
    rng = npr.RandomState(42)

    def fun(x):
        return np.max(x)

    mat = rng.randn(10, 11)
    check_grads(fun)(mat)


def test_max_axis():
    rng = npr.RandomState(42)

    def fun(x):
        return np.max(x, axis=1)

    mat = rng.randn(3, 4, 5)
    check_grads(fun)(mat)


def test_max_axis_keepdims():
    rng = npr.RandomState(42)

    def fun(x):
        return np.max(x, axis=1, keepdims=True)

    mat = rng.randn(3, 4, 5)
    check_grads(fun)(mat)


def test_min():
    rng = npr.RandomState(42)

    def fun(x):
        return np.min(x)

    mat = rng.randn(10, 11)
    check_grads(fun)(mat)


def test_min_axis():
    rng = npr.RandomState(42)

    def fun(x):
        return np.min(x, axis=1)

    mat = rng.randn(3, 4, 5)
    check_grads(fun)(mat)


def test_min_axis_keepdims():
    rng = npr.RandomState(42)

    def fun(x):
        return np.min(x, axis=1, keepdims=True)

    mat = rng.randn(3, 4, 5)
    check_grads(fun)(mat)


def test_sum_1():
    rng = npr.RandomState(42)

    def fun(x):
        return np.sum(x)

    mat = rng.randn(10, 11)
    check_grads(fun)(mat)


def test_sum_2():
    rng = npr.RandomState(42)

    def fun(x):
        return np.sum(x, axis=0)

    mat = rng.randn(10, 11)
    check_grads(fun)(mat)


def test_sum_3():
    rng = npr.RandomState(42)

    def fun(x):
        return np.sum(x, axis=0, keepdims=True)

    mat = rng.randn(10, 11)
    check_grads(fun)(mat)


def test_sum_with_axis_tuple():
    rng = npr.RandomState(42)

    def fun(x):
        return np.sum(x, axis=(1, 2))

    mat = rng.randn(10, 11, 7)
    check_grads(fun)(mat)


def test_flipud():
    rng = npr.RandomState(42)

    def fun(x):
        return np.flipud(x)

    mat = rng.randn(10, 11)
    check_grads(fun)(mat)


def test_fliplr():
    rng = npr.RandomState(42)

    def fun(x):
        return np.fliplr(x)

    mat = rng.randn(10, 11)
    check_grads(fun)(mat)


def test_rot90():
    rng = npr.RandomState(42)

    def fun(x):
        return np.rot90(x)

    mat = rng.randn(10, 11)
    check_grads(fun)(mat)


def test_cumsum_axis0():
    rng = npr.RandomState(42)

    def fun(x):
        return np.cumsum(x, axis=0)

    mat = rng.randn(10, 11)
    check_grads(fun)(mat)


def test_cumsum_axis1():
    rng = npr.RandomState(42)

    def fun(x):
        return np.cumsum(x, axis=1)

    mat = rng.randn(10, 11)
    check_grads(fun)(mat)


def test_cumsum_1d():
    rng = npr.RandomState(42)

    def fun(x):
        return np.cumsum(x)

    mat = rng.randn(10)
    check_grads(fun)(mat)


def test_cumsum_no_axis():
    rng = npr.RandomState(42)

    def fun(x):
        return np.cumsum(x)

    mat = rng.randn(10, 11)
    check_grads(fun)(mat)


def test_non_numpy_sum():
    rng = npr.RandomState(42)

    def fun(x, y):
        return sum([x, y])

    mat1 = rng.randn(10, 11)
    mat2 = rng.randn(10, 11)
    check_grads(fun)(mat1, mat2)


def test_mean_1():
    rng = npr.RandomState(42)

    def fun(x):
        return np.mean(x)

    mat = rng.randn(10, 11)
    check_grads(fun)(mat)


def test_mean_2():
    rng = npr.RandomState(42)

    def fun(x):
        return np.mean(x, axis=0)

    mat = rng.randn(10, 11)
    check_grads(fun)(mat)


def test_mean_3():
    rng = npr.RandomState(42)

    def fun(x):
        return np.mean(x, axis=0, keepdims=True)

    mat = rng.randn(10, 11)
    check_grads(fun)(mat)


def test_mean_list_of_boxes():
    assert grad(lambda x: np.mean([x, x + 2]))(0.0) == 1.0
    assert grad(lambda x: np.mean((x, x + 2)))(0.0) == 1.0


def test_std_list_of_boxes():
    # Symmetric around the mean, so grad w.r.t. x at x=0 is 0.
    assert grad(lambda x: np.std([x, x + 2]))(0.0) == 0.0
    assert grad(lambda x: np.std((x, x + 2)))(0.0) == 0.0


def test_var_list_of_boxes():
    assert grad(lambda x: np.var([x, x + 2]))(0.0) == 0.0
    assert grad(lambda x: np.var((x, x + 2)))(0.0) == 0.0


def test_index_ints():
    rng = npr.RandomState(42)
    A = rng.randn(5, 6, 4)

    def fun(x):
        return x[3, 0, 1]

    check_grads(fun)(A)


def test_index_slice():
    rng = npr.RandomState(42)
    A = rng.randn(5, 6, 4)

    def fun(x):
        return x[::-1, 2:4, :]

    check_grads(fun)(A)


def test_index_lists():
    rng = npr.RandomState(42)
    A = rng.randn(5, 6, 4)

    def fun(x):
        return x[[0, 1, 2], :, :]

    check_grads(fun)(A)


def test_index_mixed():
    rng = npr.RandomState(42)
    A = rng.randn(5, 6, 4)

    def fun(x):
        return x[3, 2:, [1, 3]]

    check_grads(fun)(A)


def test_vector_slice():
    rng = npr.RandomState(42)
    A = rng.randn(5)

    def fun(x):
        return x[2:4]

    check_grads(fun)(A)


def test_index_slice_fanout():
    rng = npr.RandomState(42)
    A = rng.randn(5, 6, 4)

    def fun(x):
        y = x[::-1, 2:4, :]
        z = x[::-1, 3:5, :]
        return y + z

    check_grads(fun)(A)


def test_index_multiple_slices():
    rng = npr.RandomState(42)
    A = rng.randn(7)

    def fun(x):
        y = x[2:6]
        z = y[1:3]
        return z

    check_grads(fun)(A)


def test_reshape_method():
    rng = npr.RandomState(42)
    A = rng.randn(5, 6, 4)

    def fun(x):
        return x.reshape((5 * 4, 6))

    check_grads(fun)(A)


def test_reshape_call():
    rng = npr.RandomState(42)
    A = rng.randn(5, 6, 4)

    def fun(x):
        return np.reshape(x, (5 * 4, 6))

    check_grads(fun)(A)


def test_reshape_method_nolist():
    rng = npr.RandomState(42)
    # The reshape can be called in two different ways:
    # like A.reshape((5,4)) or A.reshape(5,4).
    # This test checks that we support the second way.
    A = rng.randn(5, 6, 4)

    def fun(x):
        return x.reshape(5 * 4, 6)

    check_grads(fun)(A)


def test_ravel_method():
    rng = npr.RandomState(42)
    A = rng.randn(5, 6, 4)

    def fun(x):
        return x.ravel()

    check_grads(fun)(A)


def test_ravel_call():
    rng = npr.RandomState(42)
    A = rng.randn(5, 6, 4)

    def fun(x):
        return np.ravel(x)

    check_grads(fun)(A)


def test_flatten_method():
    rng = npr.RandomState(42)
    A = rng.randn(5, 6, 4)

    def fun(x):
        return x.flatten()

    check_grads(fun)(A)


def test_simple_append_list():
    A = [1.0, 2.0, 3.0]
    b = 4.0
    check_grads(np.append, argnum=(0, 1))(A, b)


def test_simple_append_arr():
    A = np.array([1.0, 2.0, 3.0])
    b = 4.0
    check_grads(np.append, argnum=(0, 1))(A, b)


def test_simple_append_list_2D():
    A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    B = [[7.0, 8.0, 9.0]]
    check_grads(np.append, argnum=(0, 1))(A, B, axis=0)


def test_simple_concatenate():
    rng = npr.RandomState(42)
    A = rng.randn(5, 6, 4)
    B = rng.randn(4, 6, 4)

    def fun(x):
        return np.concatenate((A, x))

    check_grads(fun)(B)


def test_concatenate_axis_0():
    rng = npr.RandomState(42)
    A = rng.randn(5, 6, 4)
    B = rng.randn(5, 6, 4)

    def fun(x):
        return np.concatenate((B, x, B))

    check_grads(fun)(A)


def test_concatenate_axis_1():
    rng = npr.RandomState(42)
    A = rng.randn(5, 6, 4)
    B = rng.randn(5, 6, 4)

    def fun(x):
        return np.concatenate((B, x, B), axis=1)

    check_grads(fun)(A)


def test_concatenate_axis_1_unnamed():
    """Tests whether you can specify the axis without saying "axis=1"."""
    rng = npr.RandomState(42)
    A = rng.randn(5, 6, 4)
    B = rng.randn(5, 6, 4)

    def fun(x):
        return np.concatenate((B, x, B), 1)

    check_grads(fun)(A)


def test_trace():
    rng = npr.RandomState(42)

    def fun(x):
        return np.trace(x, offset=offset)

    mat = rng.randn(10, 11)
    offset = rng.randint(-9, 11)
    check_grads(fun)(mat)


def test_trace2():
    rng = npr.RandomState(42)

    def fun(x):
        return np.trace(x, offset=offset)

    mat = rng.randn(11, 10)
    offset = rng.randint(-9, 11)
    check_grads(fun)(mat)


def test_trace_extradims():
    rng = npr.RandomState(42)

    def fun(x):
        return np.trace(x, offset=offset)

    mat = rng.randn(5, 6, 4, 3)
    offset = rng.randint(-5, 6)
    check_grads(fun)(mat)


# TODO: Allow axis1, axis2 args.
# def test_trace_extradims2():
#     def fun(x): return np.trace(x, offset=offset, axis1=3,axis2=2)
#     mat = npr.randn(5,6,4,3)
#     offset = npr.randint(-5,6)
#     check_grads(fun)(mat)


def test_diag():
    rng = npr.RandomState(42)

    def fun(x):
        return np.diag(x)

    mat = rng.randn(10, 10)
    check_grads(fun)(mat)


def test_transpose():
    rng = npr.RandomState(42)

    def fun(x):
        return x.T

    mat = rng.randn(8, 8)
    check_grads(fun)(mat)


def test_roll():
    rng = npr.RandomState(42)

    def fun(x):
        return np.roll(x, 2, axis=1)

    mat = rng.randn(4, 5)
    check_grads(fun)(mat)


def test_roll_no_axis():
    rng = npr.RandomState(42)

    def fun(x):
        return np.roll(x, 2, axis=1)

    mat = rng.randn(4, 5)
    check_grads(fun)(mat)


def test_triu():
    rng = npr.RandomState(42)

    def fun(x):
        return np.triu(x, k=2)

    mat = rng.randn(5, 5)
    check_grads(fun)(mat)


def test_tril():
    rng = npr.RandomState(42)

    def fun(x):
        return np.tril(x, k=2)

    mat = rng.randn(5, 5)
    check_grads(fun)(mat)


def test_clip():
    rng = npr.RandomState(42)

    def fun(x):
        return np.clip(x, a_min=0.1, a_max=1.1)

    mat = rng.randn(5, 5)
    check_grads(fun)(mat)


def test_prod_1():
    rng = npr.RandomState(42)

    def fun(x):
        return np.prod(x)

    mat = rng.randn(2, 3) ** 2 / 10.0 + 0.1  # Gradient unstable when zeros are present.
    check_grads(fun)(mat)


def test_prod_2():
    rng = npr.RandomState(42)

    def fun(x):
        return np.prod(x, axis=0)

    mat = rng.randn(2, 3) ** 2 + 0.1
    check_grads(fun)(mat)


def test_prod_3():
    rng = npr.RandomState(42)

    def fun(x):
        return np.prod(x, axis=0, keepdims=True)

    mat = rng.randn(2, 3) ** 2 + 0.1
    check_grads(fun)(mat)


def test_prod_4():
    rng = npr.RandomState(42)

    def fun(x):
        return np.prod(x)

    mat = rng.randn(7) ** 2 + 0.1
    check_grads(fun)(mat)


def test_1d_array():
    def fun(x):
        return np.array([x, x * 1.0, x + 2.5])

    check_grads(fun)(3.0)


def test_2d_array():
    def fun(x):
        return np.array([[x, x * 1.0, x + 2.5], [x**2, x, x / 2.0]])

    check_grads(fun)(3.0)


def test_1d_array_fanout():
    def fun(x):
        A = np.array([x, x * 1.0, x + 2.5])
        return A + A

    check_grads(fun)(3.0)


def test_2d_array_fanout():
    def fun(x):
        A = np.array([[x, x * 1.0, x + 2.5], [x**2, x, x / 2.0]])
        return A + A

    check_grads(fun)(3.0)


def test_array_from_scalar():
    def fun(x):
        return np.array(x)

    check_grads(fun)(3.0)


def test_scalar_array_box_attributes():
    # An ArrayBox-ed Python scalar is presented as a 0-dim, size-1 array
    def fun(x):
        assert x.shape == ()
        assert x.ndim == 0
        assert x.size == 1
        assert x.dtype == np.float64
        return x**2

    assert grad(fun)(3.0) == 6.0


def test_array_from_arrays():
    rng = npr.RandomState(42)

    def fun(x):
        return np.array([x, x])

    A = rng.randn(3, 2)
    check_grads(fun)(A)


def test_array_from_arrays_2():
    rng = npr.RandomState(42)

    def fun(x):
        return np.array([[2 * x, x + 1], [x, x]])

    A = rng.randn(3, 2)
    check_grads(fun)(A)


def test_len():
    rng = npr.RandomState(42)

    def fun(x):
        assert len(x) == 3
        return x

    A = rng.randn(3, 2)
    check_grads(fun)(A)


def test_r_basic():
    rng = npr.RandomState(42)
    with warnings.catch_warnings(record=True) as w:

        def fun(x):
            c = rng.randn(3, 2)
            b = np.r_[x]
            return b

        A = rng.randn(3, 2)
        check_grads(fun)(A)


def test_r_double():
    rng = npr.RandomState(42)
    with warnings.catch_warnings(record=True) as w:

        def fun(x):
            c = rng.randn(3, 2)
            b = np.r_[x, x]
            return b

        A = rng.randn(3, 2)
        check_grads(fun)(A)


def test_no_relation():
    rng = npr.RandomState(42)
    with warnings.catch_warnings(record=True) as w:
        c = rng.randn(3, 2)

        def fun(x):
            return c

        A = rng.randn(3, 2)
        check_grads(fun)(A)


def test_r_no_relation():
    rng = npr.RandomState(42)
    with warnings.catch_warnings(record=True) as w:
        c = rng.randn(3, 2)

        def fun(x):
            b = np.r_[c]
            return b

        A = rng.randn(3, 2)
        check_grads(fun)(A)


def test_r_node_and_const():
    rng = npr.RandomState(42)
    with warnings.catch_warnings(record=True) as w:
        c = rng.randn(3, 2)

        def fun(x):
            b = np.r_[x, c]
            return b

        A = rng.randn(3, 2)
        check_grads(fun)(A)


def test_r_mixed():
    rng = npr.RandomState(42)
    with warnings.catch_warnings(record=True) as w:
        c = rng.randn(3, 2)

        def fun(x):
            b = np.r_[x, c, x]
            return b

        A = rng.randn(3, 2)
        check_grads(fun)(A)


def test_r_slicing():
    rng = npr.RandomState(42)
    with warnings.catch_warnings(record=True) as w:
        c = rng.randn(10)

        def fun(x):
            b = np.r_[x, c, 1:10]
            return b

        A = rng.randn(10)
        check_grads(fun)(A)


def test_c_():
    rng = npr.RandomState(42)
    with warnings.catch_warnings(record=True) as w:
        c = rng.randn(3, 2)

        def fun(x):
            b = np.c_[x, c, x]
            return b

        A = rng.randn(3, 2)
        check_grads(fun)(A)


def test_c_mixed():
    rng = npr.RandomState(42)
    with warnings.catch_warnings(record=True) as w:
        c = rng.randn(3, 2)

        def fun(x):
            b = np.c_[x, c, x]
            return b

        A = rng.randn(3, 2)
        check_grads(fun)(A)


def test_var_ddof():
    rng = npr.RandomState(42)
    B = rng.randn(3)
    C = rng.randn(3, 4)
    D = rng.randn(1, 3)
    combo_check(np.var, (0,))([B, C, D], axis=[None], keepdims=[True, False], ddof=[0, 1])
    combo_check(np.var, (0,))([C, D], axis=[None, 1], keepdims=[True, False], ddof=[2])


def test_std_ddof():
    rng = npr.RandomState(42)
    B = rng.randn(3)
    C = rng.randn(3, 4)
    D = rng.randn(1, 3)
    combo_check(np.std, (0,))([B, C, D], axis=[None], keepdims=[True, False], ddof=[0, 1])
    combo_check(np.std, (0,))([C, D], axis=[None, 1], keepdims=[True, False], ddof=[2])


def test_where():
    rng = npr.RandomState(42)

    def fun(x, y):
        b = np.where(C, x, y)
        return b

    C = rng.randn(4, 5) > 0
    A = rng.randn(4, 5)
    B = rng.randn(4, 5)
    check_grads(fun)(A, B)


def test_squeeze_func():
    rng = npr.RandomState(42)
    A = rng.randn(5, 1, 4)

    def fun(x):
        return np.squeeze(x)

    check_grads(fun)(A)


def test_squeeze_method():
    rng = npr.RandomState(42)
    A = rng.randn(5, 1, 4)

    def fun(x):
        return x.squeeze()

    check_grads(fun)(A)


def test_repeat():
    rng = npr.RandomState(42)
    A = rng.randn(5, 3, 4)

    def fun(x):
        return np.repeat(x, 2, axis=1)

    check_grads(fun)(A)


def test_repeat_axis1_rep1():
    rng = npr.RandomState(42)
    A = rng.randn(5, 3, 4)

    def fun(x):
        return np.repeat(x, 1, axis=1)

    check_grads(fun)(A)


def test_repeat_axis0():
    rng = npr.RandomState(42)
    A = rng.randn(5, 3)

    def fun(x):
        return np.repeat(x, 2, axis=0)

    check_grads(fun)(A)


def test_repeat_1d_axis0():
    rng = npr.RandomState(42)
    A = rng.randn(5)

    def fun(x):
        return np.repeat(x, 2, axis=0)

    check_grads(fun)(A)


def test_repeat_axis0_rep1():
    rng = npr.RandomState(42)
    A = rng.randn(5, 1)

    def fun(x):
        return np.repeat(x, 1, axis=0)

    check_grads(fun)(A)


def test_expand_dims():
    rng = npr.RandomState(42)
    A = rng.randn(5, 1, 4)

    def fun(x):
        return np.expand_dims(x, 2)

    check_grads(fun)(A)


def test_tensordot_kwargs_by_position():
    def fun(x):
        return np.tensordot(x * np.ones((2, 2)), x * np.ones((2, 2)), 2)

    grad(fun)(1.0)


def test_multi_index():
    rng = npr.RandomState(42)
    A = rng.randn(3)
    fun = lambda x: np.sum(x[[0, 0]])
    check_grads(fun)(A)


def test_multi_index2():
    rng = npr.RandomState(42)
    A = rng.randn(3)
    fun = lambda x: np.sum(x[[0, 1, 0]])
    check_grads(fun)(A)


def test_index_dot_slices():
    rng = npr.RandomState(42)
    A = rng.randn(4)

    def fun(x):
        return np.dot(x[:2], x[2:])

    check_grads(fun)(A)


# def test_index_exp_slicing():
#    def fun(x):
#        b = np.index_exp[x, x]
#        return b
#    A = npr.randn(10, 1)
#    check_grads(fun)(A)

# def test_s_slicing():
#    def fun(x):
#        b = np.s_[x, x]
#        return b
#    A = npr.randn(10, 1)
#    check_grads(fun)(A)

# TODO:
# getitem


def test_cast_to_int():
    rng = npr.RandomState(42)
    inds = np.ones(5)[:, None]

    def fun(W):
        # glue W and inds together
        glued_together = np.concatenate((W, inds), axis=1)

        # separate W and inds back out
        new_W = W[:, :-1]
        new_inds = np.int64(W[:, -1])

        assert new_inds.dtype == np.int64
        return new_W[new_inds].sum()

    W = rng.randn(5, 10)
    check_grads(fun)(W)


def test_make_diagonal():
    rng = npr.RandomState(42)

    def fun(D):
        return np.make_diagonal(D, axis1=-1, axis2=-2)

    D = rng.randn(4)
    A = np.make_diagonal(D, axis1=-1, axis2=-2)
    assert np.allclose(np.diag(A), D)
    check_grads(fun)(D)

    D = rng.randn(3, 4)
    A = np.make_diagonal(D, axis1=-1, axis2=-2)
    assert all([np.allclose(np.diag(A[i]), D[i]) for i in range(3)])
    check_grads(fun)(D)


def test_diagonal():
    rng = npr.RandomState(42)

    def fun(D):
        return np.diagonal(D, axis1=-1, axis2=-2)

    D = rng.randn(4, 4)
    A = np.make_diagonal(D, axis1=-1, axis2=-2)
    check_grads(fun)(D)

    D = rng.randn(3, 4, 4)
    A = np.make_diagonal(D, axis1=-1, axis2=-2)
    check_grads(fun)(D)


def test_nan_to_num():
    rng = npr.RandomState(42)
    y = np.array([0.0, np.nan, np.inf, -np.inf])
    fun = lambda x: np.sum(np.sin(np.nan_to_num(x + y)))

    x = rng.randn(4)
    check_grads(fun)(x)


# TODO(mattjj): np.frexp returns a pair of ndarrays and the second is an int
# type, for which there is currently no vspace registered
# def test_frexp():
#    fun = lambda x: np.frexp(x)[0]
#    A = 1.2 #np.random.rand(4,3) * 0.8 + 2.1
#    check_grads(fun)(A)


def test_max_equal_values():
    def fun(x):
        return np.max(np.array([x, x]))

    check_grads(fun)(1.0)


def test_max_equal_values_2d():
    def fun(x):
        return np.max(np.array([[x, x], [x, 0.5]]), axis=1)

    check_grads(fun)(1.0)
    check_grads(fun)(-1.0)


def test_min_3_way_equality():
    def fun(x):
        return np.min(np.array([[x, x, x], [x, 0.5, 0.5], [0.5, 0.5, 0.5], [x, x, 0.5]]), axis=0)

    check_grads(fun)(1.0)
    check_grads(fun)(-1.0)


def test_maximum_equal_values():
    def fun(x):
        return np.maximum(x, x)

    check_grads(fun)(1.0)


def test_maximum_equal_values_2d():
    def fun(x):
        return np.maximum(np.array([x, x, 0.5]), np.array([[x, 0.5, x], [x, x, 0.5]]))

    check_grads(fun)(1.0)
    check_grads(fun)(-1.0)
    check_grads(fun)(2.0)


def test_linspace():
    for num in [0, 1, 5]:

        def fun(x, y):
            return np.linspace(x, y, num)

        check_grads(fun)(1.2, 3.4)
        check_grads(fun)(1.2, -3.4)
        check_grads(fun)(1.2, 1.2)


def test_astype():
    x = np.arange(3, dtype="float32")

    def f(x):
        return np.sum(np.sin(x.astype("float64")))

    assert grad(f)(x).dtype == np.dtype("float32")


def test_gradient():
    rng = npr.RandomState(42)
    check_grads(np.gradient, 0)(rng.randn(10))
    check_grads(np.gradient, 0)(rng.randn(10, 10))
    check_grads(np.gradient, 0)(rng.randn(10, 10, 10))

    for a in [None, 0, 1, -1, (0, 1), (0, -1)]:
        check_grads(np.gradient, 0)(rng.randn(10, 10, 10), axis=a)
