from __future__ import absolute_import
from __future__ import print_function
from builtins import range
import itertools as it
import autograd.numpy.random as npr
from autograd import grad
from autograd.test_util import combo_check
import warnings

def stat_check(fun, test_complex=True, **kwargs):
    # Tests functions that compute statistics, like sum, mean, etc
    x = 3.5
    A = npr.randn()
    B = npr.randn(3)
    C = npr.randn(2, 3)
    D = npr.randn(1, 3)
    check = combo_check(fun, (0,), **kwargs)
    check([x, A])
    check([B, C, D], axis=[None, 0], keepdims=[True, False])
    check([C, D], axis=[None, 0, 1], keepdims=[True, False])
    if test_complex:
        c = npr.randn() + 0.1j*npr.randn()
        E = npr.randn(2,3) + 0.1j*npr.randn(2,3)
        check([x, c, A])
        check([B, C, D, E], axis=[None, 0], keepdims=[True, False])

def unary_ufunc_check(fun, lims=[-2, 2], test_complex=True, **kwargs):
    scalar = transform(lims, 0.4)
    vector = transform(lims, npr.rand(2))
    mat    = transform(lims, npr.rand(3, 2))
    mat2   = transform(lims, npr.rand(1, 2))
    check = combo_check(fun, (0,), **kwargs)
    check([scalar, vector, mat, mat2])
    if test_complex:
        comp = transform(lims, 0.4) + 0.1j * transform(lims, 0.3)
        matc = transform(lims, npr.rand(3, 2)) + 0.1j * npr.rand(3, 2)
        check([comp, matc])

def binary_ufunc_check(fun, lims_A=[-2, 2], lims_B=[-2, 2], test_complex=True, **kwargs):
    T_A = lambda x : transform(lims_A, x)
    T_B = lambda x : transform(lims_B, x)
    scalar = 0.6
    vector = npr.rand(2)
    mat    = npr.rand(3, 2)
    mat2   = npr.rand(1, 2)
    check = combo_check(fun, (0, 1), **kwargs)
    check([T_A(scalar), T_A(vector), T_A(mat), T_A(mat2)],
          [T_B(scalar), T_B(vector), T_B(mat), T_B(mat2)])
    if test_complex:
        comp = 0.6 + 0.3j
        matc = npr.rand(3, 2) + 0.1j * npr.rand(3, 2)
        check([T_A(scalar), T_A(comp), T_A(vector), T_A(matc),  T_A(mat2)],
              [T_B(scalar), T_B(comp), T_B(vector), T_B(matc), T_B(mat2)])

def binary_ufunc_check_no_same_args(fun, lims_A=[-2, 2], lims_B=[-2, 2], test_complex=True, **kwargs):
    T_A = lambda x : transform(lims_A, x)
    T_B = lambda x : transform(lims_B, x)
    scalar1 = 0.6;   scalar2 = 0.7
    vector1 = npr.rand(2);  vector2 = npr.rand(2)
    mat11   = npr.rand(3, 2); mat12 = npr.rand(3, 2)
    mat21   = npr.rand(1, 2); mat22 = npr.rand(1, 2)
    check = combo_check(fun, (0, 1), **kwargs)
    check([T_A(scalar1), T_A(vector1), T_A(mat11), T_A(mat21)],
          [T_B(scalar2), T_B(vector2), T_B(mat12), T_B(mat22)])
    if test_complex:
        comp1 = 0.6 + 0.3j; comp2 = 0.1 + 0.2j
        matc1 = npr.rand(3, 2) + 0.1j * npr.rand(3, 2)
        matc2 = npr.rand(3, 2) + 0.1j * npr.rand(3, 2)
        check([T_A(scalar1), T_A(comp1), T_A(vector1), T_A(matc1),  T_A(mat21)],
              [T_B(scalar2), T_B(comp2), T_B(vector2), T_B(matc2), T_B(mat22)])

def transform(lims, x):
    return x * (lims[1] - lims[0]) + lims[0]
