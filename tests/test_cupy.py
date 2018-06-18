from __future__ import absolute_import
import warnings

import autograd.cupy as cp
import autograd.cupy.random as cpr
from autograd.test_util import check_grads
from autograd import grad
import pytest
import autograd.numpy as np

from autograd.test_util import combo_check

cpr.seed(1)


@pytest.mark.works
@pytest.mark.cupy
def test_dot():

    def fun(x, y):
        return cp.dot(x, y)

    mat1 = cpr.randn(10, 11)
    mat2 = cpr.randn(10, 11)
    vect1 = cpr.randn(10)
    vect2 = cpr.randn(11)
    vect3 = cpr.randn(11)

    check_grads(fun)(mat1, vect2)
    check_grads(fun)(mat1, mat2.T)
    check_grads(fun)(vect1, mat1)
    check_grads(fun)(vect2, vect3)


@pytest.mark.works
@pytest.mark.cupy
def test_dot_with_floats():
    """
    Test function for dot product.

    We are not supporting a.dot(b), so do not write this test.
    """

    def fun(x, y):
        return cp.dot(x, y)

    mat1 = cpr.randn(10, 11)
    vect1 = cpr.randn(10)
    float1 = cpr.randn()

    check_grads(fun)(mat1, float1)
    check_grads(fun)(float1, mat1)
    check_grads(fun)(vect1, float1)
    check_grads(fun)(float1, vect1)


@pytest.mark.works
@pytest.mark.cupy
def test_outer():

    def fun(x, y):
        return cp.outer(x, y)

    vect2 = cpr.randn(11)
    vect3 = cpr.randn(11)

    check_grads(fun)(vect2, vect3)
    check_grads(fun)(vect2.T, vect3)
    check_grads(fun)(vect2.T, vect3.T)


@pytest.mark.works
@pytest.mark.cupy
def test_max():

    def fun(x):
        return cp.max(x)

    mat = cpr.randn(10, 11)
    check_grads(fun)(mat)


@pytest.mark.works
@pytest.mark.cupy
def test_max_axis():

    def fun(x):
        return cp.max(x, axis=1)

    mat = cpr.randn(3, 4, 5)
    check_grads(fun)(mat)


@pytest.mark.works
@pytest.mark.cupy
def test_max_axis_keepdims():

    def fun(x):
        return cp.max(x, axis=1, keepdims=True)

    mat = cpr.randn(3, 4, 5)
    check_grads(fun)(mat)


@pytest.mark.works
@pytest.mark.cupy
def test_min():

    def fun(x):
        return cp.min(x)

    mat = cpr.randn(10, 11)
    check_grads(fun)(mat)


@pytest.mark.works
@pytest.mark.cupy
def test_min_axis():

    def fun(x):
        return cp.min(x, axis=1)

    mat = cpr.randn(3, 4, 5)
    check_grads(fun)(mat)


@pytest.mark.works
@pytest.mark.cupy
def test_min_axis_keepdims():

    def fun(x):
        return cp.min(x, axis=1, keepdims=True)

    mat = cpr.randn(3, 4, 5)
    check_grads(fun)(mat)


@pytest.mark.works
@pytest.mark.cupy
def test_sum_1():

    def fun(x):
        return cp.sum(x)

    mat = cpr.randn(10, 11)
    check_grads(fun)(mat)


@pytest.mark.works
@pytest.mark.cupy
def test_sum_2():

    def fun(x):
        return cp.sum(x, axis=0)

    mat = cpr.randn(10, 11)
    check_grads(fun)(mat)


@pytest.mark.works
@pytest.mark.cupy
def test_sum_3():

    def fun(x):
        return cp.sum(x, axis=0, keepdims=True)

    mat = cpr.randn(10, 11)
    check_grads(fun)(mat)


@pytest.mark.works
@pytest.mark.cupy
def test_sum_with_axis_tuple():

    def fun(x):
        return cp.sum(x, axis=(1, 2))

    mat = cpr.randn(10, 11, 7)
    check_grads(fun)(mat)


@pytest.mark.works
@pytest.mark.cupy
def test_flipud():

    def fun(x):
        return cp.flipud(x)

    mat = cpr.randn(10, 11)
    check_grads(fun)(mat)


@pytest.mark.works
@pytest.mark.cupy
def test_fliplr():

    def fun(x):
        return cp.fliplr(x)

    mat = cpr.randn(10, 11)
    check_grads(fun)(mat)


@pytest.mark.works
@pytest.mark.cupy
def test_rot90():

    def fun(x):
        return cp.rot90(x)

    mat = cpr.randn(10, 11)
    check_grads(fun)(mat)


@pytest.mark.works
@pytest.mark.cupy
def test_cumsum_axis0():

    def fun(x):
        return cp.cumsum(x, axis=0)

    mat = cpr.randn(10, 11)
    check_grads(fun)(mat)


@pytest.mark.works
@pytest.mark.cupy
def test_cumsum_axis1():

    def fun(x):
        return cp.cumsum(x, axis=1)

    mat = cpr.randn(10, 11)
    check_grads(fun)(mat)


@pytest.mark.works
@pytest.mark.cupy
def test_cumsum_1d():

    def fun(x):
        return cp.cumsum(x)

    mat = cpr.randn(10)
    check_grads(fun)(mat)


@pytest.mark.works
@pytest.mark.cupy
def test_cumsum_no_axis():

    def fun(x):
        return cp.cumsum(x)

    mat = cpr.randn(10, 11)
    check_grads(fun)(mat)


@pytest.mark.works
@pytest.mark.cupy
def test_non_cupy_sum():
    def fun(x, y):
        return sum([x, y])

    mat1 = cpr.randn(10, 11)
    mat2 = cpr.randn(10, 11)
    check_grads(fun)(mat1, mat2)


@pytest.mark.works
@pytest.mark.cupy
def test_mean_1():

    def fun(x):
        return cp.mean(x)

    mat = cpr.randn(10, 11)
    check_grads(fun)(mat)


@pytest.mark.works
@pytest.mark.cupy
def test_mean_2():

    def fun(x):
        return cp.mean(x, axis=0)

    mat = cpr.randn(10, 11)
    check_grads(fun)(mat)


@pytest.mark.works
@pytest.mark.cupy
def test_mean_3():

    def fun(x):
        return cp.mean(x, axis=0, keepdims=True)

    mat = cpr.randn(10, 11)
    check_grads(fun)(mat)


@pytest.mark.works
@pytest.mark.cupy
def test_index_ints():
    A = cpr.randn(5, 6, 4)

    def fun(x):
        return x[3, 0, 1]

    check_grads(fun)(A)


@pytest.mark.works
@pytest.mark.cupy
def test_index_slice():
    A = cpr.randn(5, 6, 4)

    def fun(x):
        return x[::-1, 2:4, :]

    check_grads(fun)(A)


# This test fails with scatter_add only supporting float32 dtype.
# The type of the array A below is float64.
# See comments on mut_add function for more details.
# Additional tests: `a.astype('float32')` does not pass test. Looks related
# to a precision issue.
@pytest.mark.fails_scatter_add
@pytest.mark.cupy
def test_index_lists():
    A = cpr.randn(5, 6, 4).astype('float32')

    def fun(x):
        return x[[0, 1, 2], :, :]

    check_grads(fun)(A)


@pytest.mark.fails_scatter_add
@pytest.mark.cupy
def test_index_mixed():
    A = cpr.randn(5, 6, 4).astype('float32')

    def fun(x):
        return x[3, 2:, [1, 3]]

    check_grads(fun)(A)


@pytest.mark.works
@pytest.mark.cupy
def test_vector_slice():
    A = cpr.randn(5)

    def fun(x):
        return x[2:4]

    check_grads(fun)(A)


@pytest.mark.works
@pytest.mark.cupy
def test_index_slice_fanout():
    A = cpr.randn(5, 6, 4)

    def fun(x):
        y = x[::-1, 2:4, :]
        z = x[::-1, 3:5, :]
        return y + z

    check_grads(fun)(A)


@pytest.mark.works
@pytest.mark.cupy
def test_index_multiple_slices():
    A = cpr.randn(7)

    def fun(x):
        y = x[2:6]
        z = y[1:3]
        return z

    check_grads(fun)(A)


@pytest.mark.works
@pytest.mark.cupy
def test_reshape_method():
    A = cpr.randn(5, 6, 4)

    def fun(x):
        return x.reshape((5 * 4, 6))

    check_grads(fun)(A)


@pytest.mark.works
@pytest.mark.cupy
def test_reshape_call():
    A = cpr.randn(5, 6, 4)

    def fun(x):
        return cp.reshape(x, (5 * 4, 6))

    check_grads(fun)(A)


@pytest.mark.works
@pytest.mark.cupy
def test_reshape_method_nolist():
    # The reshape can be called in two different ways:
    # like A.reshape((5,4)) or A.reshape(5,4).
    # This test checks that we support the second way.
    A = cpr.randn(5, 6, 4)

    def fun(x):
        return x.reshape(5 * 4, 6)

    check_grads(fun)(A)


@pytest.mark.works
@pytest.mark.cupy
def test_ravel_method():
    A = cpr.randn(5, 6, 4)

    def fun(x):
        return x.ravel()

    check_grads(fun)(A)


@pytest.mark.works
@pytest.mark.cupy
def test_ravel_call():
    A = cpr.randn(5, 6, 4)

    def fun(x):
        return cp.ravel(x)

    check_grads(fun)(A)


@pytest.mark.works
@pytest.mark.cupy
def test_flatten_method():
    A = cpr.randn(5, 6, 4)

    def fun(x):
        return x.flatten()

    check_grads(fun)(A)


# To keep things pure in CuPy-land (see one of the comments above)
# this should not be available as an operation to autograd-cupy users
@pytest.mark.deprecated
@pytest.mark.cupy
def test_simple_append_list():
    A = [1., 2., 3.]
    b = 4.
    check_grads(cp.append, argnum=(0, 1))(A, b)


# Deprecated; see above comment.
@pytest.mark.deprecated
@pytest.mark.cupy
def test_simple_append_arr():
    A = cp.array([1., 2., 3.])
    b = 4.
    check_grads(cp.append, argnum=(0, 1))(A, b)


# Deprecated; see above comment.
@pytest.mark.deprecated
@pytest.mark.cupy
def test_simple_append_list_2D():
    A = [[1., 2., 3.], [4., 5., 6.]]
    B = [[7., 8., 9.]]
    check_grads(cp.append, argnum=(0, 1))(A, B, axis=0)


# Fails because of some KeyError.
# I have verified that cp.concatenate works (in an IPython shell)
# What's left is probably something to do with the derivative.
# Through interactive debugging, I'm finding an error in autograd/test_util.py.
#
#     y = f(x)
#     x_vs, y_vs = vspace(x), vspace(y)
#
# In the interactive debugger, if I run f(x), then I get a segfault. If I don't run
# f(x), then inspecting y_vs, I get ArrayVSpace_{... 'dtype': dtype('O')}. This is
# very likely the cause of the segfault in my interactive debugging, as well as the
# source of the KeyError('O') that the test returns.
@pytest.mark.fail_key_error
@pytest.mark.cupy
def test_simple_concatenate():
    A = cpr.randn(5, 6, 4)
    B = cpr.randn(4, 6, 4)

    def fun(x):
        return cp.concatenate((A, x))

    check_grads(fun)(B)


# Also fails with a KeyError. Very likely related to the above test as well.
@pytest.mark.fail_key_error
@pytest.mark.cupy
def test_concatenate_axis_0():
    A = cpr.randn(5, 6, 4)
    B = cpr.randn(5, 6, 4)

    def fun(x):
        return cp.concatenate((B, x, B))

    check_grads(fun)(A)


# Also fails with a KeyError. Very likely related to the above test as well.
@pytest.mark.fail_key_error
@pytest.mark.cupy
def test_concatenate_axis_1():
    A = cpr.randn(5, 6, 4)
    B = cpr.randn(5, 6, 4)

    def fun(x):
        return cp.concatenate((B, x, B), axis=1)

    check_grads(fun)(A)


# Also fails with a KeyError. Very likely related to the above test as well.
@pytest.mark.fail_key_error
@pytest.mark.cupy
def test_concatenate_axis_1_unnamed():
    """Tests whether you can specify the axis without saying "axis=1"."""
    A = cpr.randn(5, 6, 4)
    B = cpr.randn(5, 6, 4)

    def fun(x):
        return cp.concatenate((B, x, B), 1)

    check_grads(fun)(A)


# This test uses einstein summation. CuPy does not provide a _parse_einsum_input
# function, while NumPy does. In CuPy, einsum input parsing is done
# automatically (https://github.com/cupy/cupy/blob/v4.1.0/cupy/linalg/einsum.py#L208)
# Some steps below `cupy_vjps.py` at `isinstance(operands[0], string_types)` that
# I currently do not understand.
@pytest.mark.fail_einsum
@pytest.mark.cupy
def test_trace():

    def fun(x):
        return cp.trace(x, offset=offset)

    mat = cpr.randn(10, 11)
    offset = int(cpr.randint(-9, 11))
    check_grads(fun)(mat)


@pytest.mark.fail_einsum
@pytest.mark.cupy
def test_trace2():

    def fun(x):
        return cp.trace(x, offset=offset)

    mat = cpr.randn(11, 10)
    offset = int(cpr.randint(-9, 11))
    check_grads(fun)(mat)


@pytest.mark.fail_einsum
@pytest.mark.cupy
def test_trace_extradims():

    def fun(x):
        return cp.trace(x, offset=offset)

    mat = cpr.randn(5, 6, 4, 3)
    offset = int(cpr.randint(-5, 6))
    check_grads(fun)(mat)


# TODO: Allow axis1, axis2 args.
# def test_trace_extradims2():
#     def fun(x): return cp.trace(x, offset=offset, axis1=3,axis2=2)
#     mat = cpr.randn(5,6,4,3)
#     offset = cpr.randint(-5,6)
#     check_grads(fun)(mat)


@pytest.mark.works
@pytest.mark.cupy
def test_diag():

    def fun(x):
        return cp.diag(x)

    mat = cpr.randn(10, 10)
    check_grads(fun)(mat)


@pytest.mark.works
@pytest.mark.cupy
def test_transpose():

    def fun(x):
        return x.T

    mat = cpr.randn(8, 8)
    check_grads(fun)(mat)


@pytest.mark.works
@pytest.mark.cupy
def test_roll():

    def fun(x):
        return cp.roll(x, 2, axis=1)

    mat = cpr.randn(4, 5)
    check_grads(fun)(mat)


@pytest.mark.works
@pytest.mark.cupy
def test_roll_no_axis():

    def fun(x):
        return cp.roll(x, 2, axis=1)

    mat = cpr.randn(4, 5)
    check_grads(fun)(mat)


@pytest.mark.works
@pytest.mark.cupy
def test_triu():

    def fun(x):
        return cp.triu(x, k=2)

    mat = cpr.randn(5, 5)
    check_grads(fun)(mat)


@pytest.mark.works
@pytest.mark.cupy
def test_tril():

    def fun(x):
        return cp.tril(x, k=2)

    mat = cpr.randn(5, 5)
    check_grads(fun)(mat)


@pytest.mark.works
@pytest.mark.cupy
def test_clip():

    def fun(x):
        return cp.clip(x, a_min=0.1, a_max=1.1)

    mat = cpr.randn(5, 5)
    check_grads(fun)(mat)


@pytest.mark.works
@pytest.mark.cupy
def test_prod_1():

    def fun(x):
        return cp.prod(x)

    mat = cpr.randn(2, 3) ** 2 / 10.0 + 0.1  # Gradient unstable when zeros are present.
    check_grads(fun)(mat)



@pytest.mark.works
@pytest.mark.cupy
def test_prod_2():

    def fun(x):
        return cp.prod(x, axis=0)

    mat = cpr.randn(2, 3) ** 2 + 0.1
    check_grads(fun)(mat)


@pytest.mark.works
@pytest.mark.cupy
def test_prod_3():

    def fun(x):
        return cp.prod(x, axis=0, keepdims=True)

    mat = cpr.randn(2, 3) ** 2 + 0.1
    check_grads(fun)(mat)


@pytest.mark.works
@pytest.mark.cupy
def test_prod_4():

    def fun(x):
        return cp.prod(x)

    mat = cpr.randn(7) ** 2 + 0.1
    check_grads(fun)(mat)


# This test below returns an "unsupported dtype object" error.
# To debug this, I went into the CuPy wrapper (`cupy_wrapper.py`) and added in a PDB
# trace step. There, I was able to inspect the inputs to the array_from_args function.
#
# In the forward computation, we get a list of floats [3.0, 3.0, 3.0] as the input array.
# In the backward computation, we get a list of `ArrayBox`es. ArrayBoxes cannot be passed
# into the CuPy constructor. I have attempted to modify cupy_wrapper.array, but after
# finding out how hairy it would be to modify the function to automatically convert the
# list of boxes into a list of floats (rather than a list of single element CuPy arrays),
# I gave up. I think there has to be a better way to approach this.
@pytest.mark.fail_dtype_object
@pytest.mark.cupy
def test_1d_array():

    def fun(x):
        # return cp.array([x, x * 1.0, x + 2.5])
        return cp.array([x, x, x])

    check_grads(fun)(3.0)


@pytest.mark.fail_dtype_object
@pytest.mark.cupy
def test_2d_array():

    def fun(x):
        return cp.array([[x, x * 1.0, x + 2.5], [x ** 2, x, x / 2.0]])

    check_grads(fun)(3.0)


@pytest.mark.fail_dtype_object
@pytest.mark.cupy
def test_1d_array_fanout():

    def fun(x):
        A = cp.array([x, x * 1.0, x + 2.5])
        return A + A

    check_grads(fun)(3.0)


@pytest.mark.fail_dtype_object
@pytest.mark.cupy
def test_2d_array_fanout():

    def fun(x):
        A = cp.array([[x, x * 1.0, x + 2.5], [x ** 2, x, x / 2.0]])
        return A + A

    check_grads(fun)(3.0)


@pytest.mark.works
@pytest.mark.cupy
def test_array_from_scalar():

    def fun(x):
        return cp.array(x)

    check_grads(fun)(3.0)


# I believe that this is unsupported behaviour in CuPy: we cannot pass in a list of arrays
# as this would entail implicit data movement from CPU to GPU.
@pytest.mark.fail_dtype_object
@pytest.mark.cupy
def test_array_from_arrays():

    def fun(x):
        return cp.array([x, x])

    A = cpr.randn(3, 2)
    check_grads(fun)(A)


# Also suspect this test fails because of unsupported CuPy behaviour.
@pytest.mark.fail_dtype_object
@pytest.mark.cupy
def test_array_from_arrays_2():

    def fun(x):
        return cp.array([[2 * x, x + 1], [x, x]])

    A = cpr.randn(3, 2)
    check_grads(fun)(A)


@pytest.mark.works
@pytest.mark.cupy
def test_len():

    def fun(x):
        assert len(x) == 3
        return x

    A = cpr.randn(3, 2)
    check_grads(fun)(A)


# I do not have a hypothesis as to why this one is failing.
@pytest.mark.fail_dtype_object
@pytest.mark.cupy
def test_r_basic():
    with warnings.catch_warnings(record=True) as w:

        def fun(x):
            c = cpr.randn(3, 2)
            b = cp.r_[x]
            return b

        A = cpr.randn(3, 2)
        check_grads(fun)(A)


# Also not sure why this one fails.
@pytest.mark.fail_dtype_object
@pytest.mark.cupy
def test_r_double():
    with warnings.catch_warnings(record=True) as w:

        def fun(x):
            c = cpr.randn(3, 2)
            b = cp.r_[x, x]
            return b

        A = cpr.randn(3, 2)
        check_grads(fun)(A)


@pytest.mark.works
@pytest.mark.cupy
def test_no_relation():
    with warnings.catch_warnings(record=True) as w:
        c = cpr.randn(3, 2)

        def fun(x):
            return c

        A = cpr.randn(3, 2)
        check_grads(fun)(A)


@pytest.mark.works
@pytest.mark.cupy
def test_r_no_relation():
    with warnings.catch_warnings(record=True) as w:
        c = cpr.randn(3, 2)

        def fun(x):
            b = cp.r_[c]
            return b

        A = cpr.randn(3, 2)
        check_grads(fun)(A)


# Probably same general issue as other ones that involve cp._r
@pytest.mark.fail_dtype_object
@pytest.mark.cupy
def test_r_node_and_const():
    with warnings.catch_warnings(record=True) as w:
        c = cpr.randn(3, 2)

        def fun(x):
            b = cp.r_[x, c]
            return b

        A = cpr.randn(3, 2)
        check_grads(fun)(A)


# Probably failing because of cp._r
@pytest.mark.fail_dtype_object
@pytest.mark.cupy
def test_r_mixed():
    with warnings.catch_warnings(record=True) as w:
        c = cpr.randn(3, 2)

        def fun(x):
            b = cp.r_[x, c, x]
            return b

        A = cpr.randn(3, 2)
        check_grads(fun)(A)


# Fails because of NotImplementedError error raised.
@pytest.mark.fail_not_implemented
@pytest.mark.cupy
def test_r_slicing():
    with warnings.catch_warnings(record=True) as w:
        c = cpr.randn(10)

        def fun(x):
            b = cp.r_[x, c, 1:10]
            return b

        A = cpr.randn(10)
        check_grads(fun)(A)


@pytest.mark.fail_dtype_object
@pytest.mark.cupy
def test_c_():
    with warnings.catch_warnings(record=True) as w:
        c = cpr.randn(3, 2)

        def fun(x):
            b = cp.c_[x, c, x]
            return b

        A = cpr.randn(3, 2)
        check_grads(fun)(A)


@pytest.mark.fail_dtype_object
@pytest.mark.cupy
def test_c_mixed():
    with warnings.catch_warnings(record=True) as w:
        c = cpr.randn(3, 2)

        def fun(x):
            b = cp.c_[x, c, x]
            return b

        A = cpr.randn(3, 2)
        check_grads(fun)(A)


@pytest.mark.works
@pytest.mark.cupy
def test_var_ddof():
    B = cpr.randn(3)
    C = cpr.randn(3, 4)
    D = cpr.randn(1, 3)
    combo_check(cp.var, (0,))(
        [B, C, D], axis=[None], keepdims=[True, False], ddof=[0, 1]
    )
    combo_check(cp.var, (0,))([C, D], axis=[None, 1], keepdims=[True, False], ddof=[2])


@pytest.mark.works
@pytest.mark.cupy
def test_std_ddof():
    B = cpr.randn(3)
    C = cpr.randn(3, 4)
    D = cpr.randn(1, 3)
    combo_check(cp.std, (0,))(
        [B, C, D], axis=[None], keepdims=[True, False], ddof=[0, 1]
    )
    combo_check(cp.std, (0,))([C, D], axis=[None, 1], keepdims=[True, False], ddof=[2])


@pytest.mark.works
@pytest.mark.cupy
def test_where():

    def fun(x, y):
        b = cp.where(C, x, y)
        return b

    C = cpr.randn(4, 5) > 0
    A = cpr.randn(4, 5)
    B = cpr.randn(4, 5)
    check_grads(fun)(A, B)


@pytest.mark.works
@pytest.mark.cupy
def test_squeeze_func():
    A = cpr.randn(5, 1, 4)

    def fun(x):
        return cp.squeeze(x)

    check_grads(fun)(A)


@pytest.mark.works
@pytest.mark.cupy
def test_squeeze_method():
    A = cpr.randn(5, 1, 4)

    def fun(x):
        return x.squeeze()

    check_grads(fun)(A)


@pytest.mark.works
@pytest.mark.cupy
def test_repeat():
    A = cpr.randn(5, 3, 4)

    def fun(x):
        return cp.repeat(x, 2, axis=1)

    check_grads(fun)(A)


@pytest.mark.works
@pytest.mark.cupy
def test_repeat_axis1_rep1():
    A = cpr.randn(5, 3, 4)

    def fun(x):
        return cp.repeat(x, 1, axis=1)

    check_grads(fun)(A)


@pytest.mark.works
@pytest.mark.cupy
def test_repeat_axis0():
    A = cpr.randn(5, 3)

    def fun(x):
        return cp.repeat(x, 2, axis=0)

    check_grads(fun)(A)


@pytest.mark.works
@pytest.mark.cupy
def test_repeat_1d_axis0():
    A = cpr.randn(5)

    def fun(x):
        return cp.repeat(x, 2, axis=0)

    check_grads(fun)(A)


@pytest.mark.works
@pytest.mark.cupy
def test_repeat_axis0_rep1():
    A = cpr.randn(5, 1)

    def fun(x):
        return cp.repeat(x, 1, axis=0)

    check_grads(fun)(A)


@pytest.mark.works
@pytest.mark.cupy
def test_expand_dims():
    A = cpr.randn(5, 1, 4)

    def fun(x):
        return cp.expand_dims(x, 2)

    check_grads(fun)(A)


# Error: AttributeError: 'float' object has no attribute 'shape'.
@pytest.mark.fail_ndim
@pytest.mark.cupy
def test_tensordot_kwargs_by_position():

    def fun(x):
        return cp.tensordot(x * cp.ones((2, 2)), x * cp.ones((2, 2)), 2)

    grad(fun)(1.0)


# Fails because of scatter_add only supporting int32, float32, uint32, uint64 as data type.
@pytest.mark.fail_scatter_add
@pytest.mark.cupy
def test_multi_index():
    A = cpr.randn(3)
    fun = lambda x: cp.sum(x[[0, 0]])
    check_grads(fun)(A)


@pytest.mark.fail_scatter_add
@pytest.mark.cupy
def test_multi_index2():
    A = cpr.randn(3)
    fun = lambda x: cp.sum(x[[0, 1, 0]])
    check_grads(fun)(A)


@pytest.mark.works
@pytest.mark.cupy
def test_index_dot_slices():
    A = cpr.randn(4)

    def fun(x):
        return cp.dot(x[:2], x[2:])

    check_grads(fun)(A)


# def test_index_exp_slicing():
#    def fun(x):
#        b = cp.index_exp[x, x]
#        return b
#    A = cpr.randn(10, 1)
#    check_grads(fun)(A)

# def test_s_slicing():
#    def fun(x):
#        b = cp.s_[x, x]
#        return b
#    A = cpr.randn(10, 1)
#    check_grads(fun)(A)

# TODO:
# getitem


# Error: ValueError: object __array__ method not producing an array
@pytest.mark.fail_array
@pytest.mark.cupy
def test_cast_to_int():
    inds = cp.ones(5)[:, None]

    def fun(W):
        # glue W and inds together
        glued_together = cp.concatenate((W, inds), axis=1)

        # separate W and inds back out
        new_W = W[:, :-1]
        new_inds = cp.int64(W[:, -1])

        assert new_inds.dtype == cp.int64
        return new_W[new_inds].sum()

    W = cp.random.randn(5, 10)
    check_grads(fun)(W)


# Error: ValueError: object __array__ method not producing an array
@pytest.mark.fail_array
@pytest.mark.cupy
def test_make_diagonal():

    def fun(D):
        return cp.make_diagonal(D, axis1=-1, axis2=-2)

    D = cp.random.randn(4)
    A = cp.make_diagonal(D, axis1=-1, axis2=-2)
    assert np.allclose(cp.diag(A), D)
    check_grads(fun)(D)

    D = cp.random.randn(3, 4)
    A = cp.make_diagonal(D, axis1=-1, axis2=-2)
    assert all([np.allclose(cp.diag(A[i]), D[i]) for i in range(3)])
    check_grads(fun)(D)


@pytest.mark.works
@pytest.mark.cupy
def test_diagonal():

    def fun(D):
        return cp.diagonal(D, axis1=-1, axis2=-2)

    D = cp.random.randn(4, 4)
    A = cp.make_diagonal(D, axis1=-1, axis2=-2)
    check_grads(fun)(D)

    D = cp.random.randn(3, 4, 4)
    A = cp.make_diagonal(D, axis1=-1, axis2=-2)
    check_grads(fun)(D)


# CuPy does not have a nan_to_num implemented at the moment.
@pytest.mark.deprecated
@pytest.mark.cupy
def test_nan_to_num():
    y = cp.array([0., cp.nan, cp.inf, -cp.inf])
    fun = lambda x: cp.sum(cp.sin(cp.nan_to_num(x + y)))

    x = cp.random.randn(4)
    check_grads(fun)(x)


# TODO(mattjj): cp.frexp returns a pair of ndarrays and the second is an int
# type, for which there is currently no vspace registered
# def test_frexp():
#    fun = lambda x: cp.frexp(x)[0]
#    A = 1.2 #cp.random.rand(4,3) * 0.8 + 2.1
#    check_grads(fun)(A)


# Error: ValueError: Unsupported dtype object
@pytest.mark.fail_dtype_object
@pytest.mark.cupy
def test_max_equal_values():

    def fun(x):
        return cp.max(cp.array([x, x]))

    check_grads(fun)(1.0)


# Error: ValueError: Unsupported dtype object
@pytest.mark.fail_dtype_object
@pytest.mark.cupy
def test_max_equal_values_2d():

    def fun(x):
        return cp.max(cp.array([[x, x], [x, 0.5]]), axis=1)

    check_grads(fun)(1.0)
    check_grads(fun)(-1.0)


# Error: ValueError: Unsupported dtype object
@pytest.mark.fail_dtype_object
@pytest.mark.cupy
def test_min_3_way_equality():

    def fun(x):
        return cp.min(
            cp.array([[x, x, x], [x, 0.5, 0.5], [0.5, 0.5, 0.5], [x, x, 0.5]]), axis=0
        )

    check_grads(fun)(1.0)
    check_grads(fun)(-1.0)


# AttributeError: 'float' object has no attribute 'shape'
@pytest.mark.fail_attribute_error
@pytest.mark.cupy
def test_maximum_equal_values():

    def fun(x):
        return cp.maximum(x, x)

    check_grads(fun)(1.0)


# ValueError: Unsupported dtype object
@pytest.mark.fail_dtype_object
@pytest.mark.cupy
def test_maximum_equal_values_2d():

    def fun(x):
        return cp.maximum(cp.array([x, x, 0.5]), cp.array([[x, 0.5, x], [x, x, 0.5]]))

    check_grads(fun)(1.0)
    check_grads(fun)(-1.0)
    check_grads(fun)(2.0)


@pytest.mark.works
@pytest.mark.cupy
def test_linspace():
    for num in [0, 1, 5]:

        def fun(x, y):
            return cp.linspace(x, y, num)

        check_grads(fun)(1.2, 3.4)
        check_grads(fun)(1.2, -3.4)
        check_grads(fun)(1.2, 1.2)


# Commenting out this test because CuPy does not support type casting.
# @pytest.mark.cupy
# def test_astype():
#     x = cp.arange(3, dtype="float32")
#
#     def f(x):
#         return cp.sum(cp.sin(x.astype("float64")))
#
#     assert grad(f)(x).dtype == cp.dtype("float32")
