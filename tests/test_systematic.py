from __future__ import absolute_import
import autograd.numpy.random as npr
import autograd.numpy as np
import operator as op
from numpy_utils import (combo_check, stat_check, unary_ufunc_check,
                         binary_ufunc_check, binary_ufunc_check_no_same_args)
npr.seed(0)

# Array statistics functions
def test_max():  stat_check(np.max)
def test_all():  stat_check(np.all)
def test_any():  stat_check(np.any)
def test_max():  stat_check(np.max)
def test_mean(): stat_check(np.mean)
def test_min():  stat_check(np.min)
def test_sum():  stat_check(np.sum)
def test_prod(): stat_check(np.prod)
def test_var():  stat_check(np.var)
def test_std():  stat_check(np.std)

# Unary ufunc tests

def test_sin():     unary_ufunc_check(np.sin)
def test_abs():     unary_ufunc_check(np.abs, lims=[0.1, 4.0])
def test_absolute():unary_ufunc_check(np.absolute, lims=[0.1, 4.0])
def test_arccosh(): unary_ufunc_check(np.arccosh, lims=[1.1, 4.0])
def test_arcsinh(): unary_ufunc_check(np.arcsinh, lims=[-0.9, 0.9])
def test_arctanh(): unary_ufunc_check(np.arctanh, lims=[-0.9, 0.9])
def test_ceil():    unary_ufunc_check(np.ceil, lims=[-1.5, 1.5], test_complex=False)
def test_cos():     unary_ufunc_check(np.cos)
def test_cosh():    unary_ufunc_check(np.cosh)
def test_deg2rad(): unary_ufunc_check(np.deg2rad, test_complex=False)
def test_degrees(): unary_ufunc_check(lambda x : np.degrees(x)/50.0, test_complex=False)
def test_exp():     unary_ufunc_check(np.exp)
def test_exp2():    unary_ufunc_check(np.exp2)
def test_expm1():   unary_ufunc_check(np.expm1)
def test_fabs():    unary_ufunc_check(np.fabs, test_complex=False)
def test_floor():   unary_ufunc_check(np.floor, lims=[-1.5, 1.5], test_complex=False)
def test_log():     unary_ufunc_check(np.log,   lims=[0.2, 2.0])
def test_log10():   unary_ufunc_check(np.log10, lims=[0.2, 2.0])
def test_log1p():   unary_ufunc_check(np.log1p, lims=[0.2, 2.0])
def test_log2():    unary_ufunc_check(np.log2,  lims=[0.2, 2.0])
def test_rad2deg(): unary_ufunc_check(lambda x : np.rad2deg(x)/50.0, test_complex=False)
def test_radians(): unary_ufunc_check(np.radians, test_complex=False)
def test_sign():    unary_ufunc_check(np.sign)
def test_sin():     unary_ufunc_check(np.sin)
def test_sinh():    unary_ufunc_check(np.sinh)
def test_sqrt():    unary_ufunc_check(np.sqrt, lims=[1.0, 3.0])
def test_square():  unary_ufunc_check(np.square, test_complex=False)
def test_tan():     unary_ufunc_check(np.tan, lims=[-1.1, 1.1])
def test_tanh():    unary_ufunc_check(np.tanh)
def test_real():    unary_ufunc_check(np.real)
def test_real_ic(): unary_ufunc_check(np.real_if_close)
def test_imag():    unary_ufunc_check(np.imag)
def test_conj():    unary_ufunc_check(np.conj)
def test_angle():   unary_ufunc_check(np.angle)

# Binary ufunc tests

def test_add(): binary_ufunc_check(np.add)
def test_logaddexp(): binary_ufunc_check(np.logaddexp, test_complex=False)
def test_logaddexp2(): binary_ufunc_check(np.logaddexp2, test_complex=False)
def test_remainder(): binary_ufunc_check_no_same_args(np.remainder, lims_A=[-0.9, 0.9], lims_B=[0.7, 1.9], test_complex=False)
def test_true_divide(): binary_ufunc_check(np.true_divide, lims_B=[0.8, 1.2], test_complex=False)
def test_mod(): binary_ufunc_check_no_same_args(np.mod,    lims_B=[0.8, 2.1], test_complex=False)
def test_true_divide_neg(): binary_ufunc_check(np.true_divide, lims_B=[-0.3, -2.0], test_complex=False)
def test_mod_neg(): binary_ufunc_check_no_same_args(np.mod,    lims_B=[-0.3, -2.0], test_complex=False)

def test_op_mul(): binary_ufunc_check(op.mul)
def test_op_add(): binary_ufunc_check(op.add)
def test_op_sub(): binary_ufunc_check(op.sub)
def test_op_mod(): binary_ufunc_check_no_same_args(op.mod, lims_B=[0.3, 2.0], test_complex=False)
def test_op_mod_neg(): binary_ufunc_check_no_same_args(op.mod, lims_B=[-0.3, -2.0], test_complex=False)



# Misc tests

R = npr.randn
def test_transpose(): combo_check(np.transpose, [0],
                                  [R(2, 3, 4)], axes = [None, [0, 1, 2], [0, 2, 1],
                                                              [2, 0, 1], [2, 1, 0],
                                                              [1, 0, 2], [1, 2, 0]])
def test_repeat(): combo_check(np.repeat, [0], [R(2, 3, 4), R(3, 1)],
                               repeats=[0,1,2], axis = [None, 0, 1])

def test_diff():
    combo_check(np.diff, [0], [R(5,5), R(5,5,5)], n=[1,2], axis=[0,1])
    combo_check(np.diff, [0], [R(1), R(1,1)], axis=[0])
    combo_check(np.diff, [0], [R(1,1), R(3,1)], axis=[1])

def test_tile():
    combo_check(np.tile, [0], [R(2,1,3,1)], reps=[(1, 4, 1, 2)])
    combo_check(np.tile, [0], [R(1,2)], reps=[(1,2), (2,3), (3,2,1)])
    combo_check(np.tile, [0], [R(1)], reps=[(2,), 2])

def test_kron():
    combo_check(np.kron, [0,1], [R(5,5), R(4,4)], [R(3,3), R(2,2)])

def test_inner(): combo_check(np.inner, [0, 1],
                            [1.5, R(3), R(2, 3)],
                            [0.3, R(3), R(4, 3)])
def test_dot(): combo_check(np.dot, [0, 1],
                            [1.5, R(3), R(2, 3), R(2, 2, 3)],
                            [0.3, R(3), R(3, 4), R(2, 3, 4)])
def test_matmul(): combo_check(np.matmul, [0, 1],
                               [R(3), R(2, 3), R(2, 2, 3)],
                               [R(3), R(3, 4), R(2, 3, 4)])
def test_tensordot_1(): combo_check(np.tensordot, [0, 1],
                                    [R(1, 3), R(2, 3, 2)],
                                    [R(3),    R(3, 1),    R(3, 4, 2)],
                                    axes=[ [(1,), (0,)] ])
def test_tensordot_2(): combo_check(np.tensordot, [0, 1],
                                    [R(3),    R(3, 1),    R(3, 4, 2)],
                                    [R(1, 3), R(2, 3, 2)],
                                    axes=[ [(0,), (1,)] ])
def test_tensordot_3(): combo_check(np.tensordot, [0, 1],
                                    [R(2, 3),    R(2, 3, 4)],
                                    [R(1, 2, 3), R(2, 2, 3, 4)],
                                    axes=[ [(0, 1), (1, 2)] ,  [(1, 0), (2, 1)] ])
def test_tensordot_4(): combo_check(np.tensordot, [0, 1],
                                    [R(2, 2), R(4, 2, 2)],
                                    [R(2, 2), R(2, 2, 4)],
                                    axes=[1, 2])
def test_tensordot_5(): combo_check(np.tensordot, [0, 1], [R(4)], [R()], axes=[0])
def test_tensordot_6(): combo_check(np.tensordot, [0, 1], [R(2,6)], [R(6,3)], axes=[[[-1], [0]]])

# Need custom tests because gradient is undefined when arguments are identical.
def test_maximum(): combo_check(np.maximum, [0, 1],
                               [R(1), R(1,4), R(3, 4)],
                               [R(1), R(1,4), R(3, 4)])
def test_fmax(): combo_check(np.fmax, [0, 1],
                            [R(1), R(1,4), R(3, 4)],
                            [R(1), R(1,4), R(3, 4)])

def test_minimum(): combo_check(np.minimum, [0, 1],
                               [R(1), R(1,4), R(3, 4)],
                               [R(1), R(1,4), R(3, 4)])
def test_fmin(): combo_check(np.fmin, [0, 1],
                            [R(1), R(1,4), R(3, 4)],
                            [R(1), R(1,4), R(3, 4)])

def test_sort():       combo_check(np.sort, [0], [R(1), R(7)])
def test_msort():     combo_check(np.msort, [0], [R(1), R(7)])
def test_partition(): combo_check(np.partition, [0],
                                  [R(7), R(14)], kth=[0, 3, 6])

def test_atleast_1d(): combo_check(np.atleast_1d, [0], [1.2, R(1), R(7), R(1,4), R(2,4), R(2, 4, 5)])
def test_atleast_2d(): combo_check(np.atleast_2d, [0], [1.2, R(1), R(7), R(1,4), R(2,4), R(2, 4, 5)])
def test_atleast_3d(): combo_check(np.atleast_3d, [0], [1.2, R(1), R(7), R(1,4), R(2,4), R(2, 4, 5),
                                                        R(2, 4, 3, 5)])

def test_einsum_transpose():  combo_check(np.einsum, [1],    ['ij->ji'], [R(1, 1), R(4,4), R(3,4)])
def test_einsum_matmult():    combo_check(np.einsum, [1, 2], ['ij,jk->ik'], [R(2, 3)], [R(3,4)])
def test_einsum_matmult_broadcast(): combo_check(np.einsum, [1, 2], ['...ij,...jk->...ik'],
                                                 [R(2, 3), R(2, 2, 3)],
                                                 [R(3, 4), R(2, 3, 4)])
def test_einsum_covsum():     combo_check(np.einsum, [1, 2], ['ijk,lji->lki'], [R(3, 4, 4)], [R(4, 4, 3)])
def test_einsum_ellipses(): combo_check(np.einsum, [1, 2], ['...jk,...lj->...lk', '...,...->...'],
                                        [R(4, 4), R(3, 4, 4)],
                                        [R(4, 4), R(3, 4, 4)])
def test_einsum_ellipses_tail(): combo_check(np.einsum, [1, 2], ['jk...,lj...->lk...'],
                                             [R(3, 2), R(3, 2, 4)],
                                             [R(2, 3), R(2, 3, 4)])
def test_einsum_ellipses_center(): combo_check(np.einsum, [1, 2], ['j...k,lj...->lk...'],
                                               [R(2, 2), R(2, 2, 2)],
                                               [R(2, 2), R(2, 2, 2)])
def test_einsum_three_args(): combo_check(np.einsum, [1, 2], ['ijk,lji,lli->lki'],
                                          [R(3, 4, 4)], [R(4, 4, 3)], [R(4, 4, 3)])

def test_einsum2_transpose():  combo_check(np.einsum, [0], [R(1, 1), R(4,4), R(3,4)], [(0,1)], [(1,0)])
def test_einsum2_matmult():    combo_check(np.einsum, [0, 2], [R(2, 3)], [(0,1)], [R(3,4)], [(1,2)], [(0,2)])
def test_einsum2_matmult_broadcast(): combo_check(np.einsum, [0, 2],
                                                  [R(2, 3), R(2, 2, 3)], [(Ellipsis, 0, 1)],
                                                  [R(3, 4), R(2, 3, 4)], [(Ellipsis, 1, 2)],
                                                  [(Ellipsis, 0, 2)])
def test_einsum2_covsum():     combo_check(np.einsum, [0, 2], [R(3, 4, 4)], [(0,1,2)], [R(4, 4, 3)], [(3,1,0)], [(3,2,0)])
def test_einsum2_three_args(): combo_check(np.einsum, [0, 2],
                                          [R(3, 4, 4)], [(0,1,2)], [R(4, 4, 3)], [(3,1,0)], [R(4, 4, 3)], [(3,3,0)], [(3,2,0)])

def test_trace():    combo_check(np.trace, [0], [R(5, 5), R(4, 5), R(5, 4), R(3, 4, 5)], offset=[-1, 0, 1])
def test_diag():     combo_check(np.diag, [0], [R(5, 5)], k=[-1, 0, 1])
def test_diag_flat():combo_check(np.diag, [0], [R(5)],    k=[-1, 0, 1])
def test_tril():     combo_check(np.tril, [0], [R(5, 5)], k=[-1, 0, 1])
def test_triu():     combo_check(np.tril, [0], [R(5, 5)], k=[-1, 0, 1])
def test_tril_3d():  combo_check(np.tril, [0], [R(5, 5, 4)], k=[-1, 0, 1])
def test_triu_3d():  combo_check(np.tril, [0], [R(5, 5, 4)], k=[-1, 0, 1])

def test_swapaxes(): combo_check(np.swapaxes, [0], [R(3, 4, 5)], axis1=[0, 1, 2], axis2=[0, 1, 2])
def test_rollaxis(): combo_check(np.rollaxis, [0], [R(2, 3, 4)], axis =[0, 1, 2], start=[0, 1, 2, 3])
def test_cross():    combo_check(np.cross, [0, 1], [R(3, 3)], [R(3, 3)],
                                 axisa=[-1, 0, 1], axisb=[-1, 0, 1], axisc=[-1, 0, 1], axis=[None, -1, 0, 1])

def test_vsplit_2d(): combo_check(np.vsplit, [0], [R(4, 8)],    [4, [1, 2]])
def test_vsplit_3d(): combo_check(np.vsplit, [0], [R(4, 4, 4)], [2, [1, 2]])
def test_hsplit_2d(): combo_check(np.hsplit, [0], [R(4, 8)],    [4, [1, 2]])
def test_hsplit_3d(): combo_check(np.hsplit, [0], [R(4, 4, 4)], [2, [1, 2]])
def test_dsplit_3d(): combo_check(np.dsplit, [0], [R(4, 4, 4)], [2, [1, 2]])

def test_split_1d(): combo_check(np.split, [0], [R(1), R(7)], [1],         axis=[0])
def test_split_2d(): combo_check(np.split, [0], [R(4, 8)],    [4, [1, 2]], axis=[0, 1])
def test_split_3d(): combo_check(np.split, [0], [R(4, 4, 4)], [2, [1, 2]], axis=[0, 1, 2])

def test_array_split_1d(): combo_check(np.array_split, [0], [R(1), R(7)], [1, 3],      axis=[0])
def test_array_split_2d(): combo_check(np.array_split, [0], [R(7, 7)],    [4, [3, 5]], axis=[0, 1])
def test_array_split_3d(): combo_check(np.array_split, [0], [R(7, 7, 7)], [4, [3, 5]], axis=[0, 1, 2])

def test_concatenate_1ist():  combo_check(np.concatenate, [0], [(R(1), R(3))],             axis=[0])
def test_concatenate_tuple(): combo_check(np.concatenate, [0], [[R(1), R(3)]],             axis=[0])
def test_concatenate_2d():    combo_check(np.concatenate, [0], [(R(2, 2), R(2, 2))],       axis=[0, 1])
def test_concatenate_3d():    combo_check(np.concatenate, [0], [(R(2, 2, 2), R(2, 2, 2))], axis=[0, 1, 2])

def test_vstack_1d(): combo_check(np.vstack, [0], [R(2), (R(2), R(2))])
def test_vstack_2d(): combo_check(np.vstack, [0], [R(2, 3), (R(2, 4), R(1, 4))])
def test_vstack_3d(): combo_check(np.vstack, [0], [R(2, 3, 4), (R(2, 3, 4), R(5, 3, 4))])
def test_hstack_1d(): combo_check(np.hstack, [0], [R(2), (R(2), R(2))])
def test_hstack_2d(): combo_check(np.hstack, [0], [R(3, 2), (R(3, 4), R(3, 5))])
def test_hstack_3d(): combo_check(np.hstack, [0], [R(2, 3, 4), (R(2, 1, 4), R(2, 5, 4))])

def test_stack_1d():  combo_check(np.stack,  [0], [(R(2),), (R(2), R(2))], axis=[0, 1])

def test_row_stack_1d(): combo_check(np.row_stack, [0], [R(2), (R(2), R(2))])
def test_row_stack_2d(): combo_check(np.row_stack, [0], [R(2, 3), (R(2, 4), R(1, 4))])
def test_column_stack_1d(): combo_check(np.column_stack, [0], [R(2), (R(2), R(2))])
def test_column_stack_2d(): combo_check(np.column_stack, [0], [R(2, 2), (R(2, 2), R(2, 2))])

def test_select(): combo_check(np.select, [1], [[R(3,4,5) > 0, R(3,4,5) > 0, R(3,4,5) > 0]],
                                               [[R(3,4,5),     R(3,4,5),     R(3,4,5)]], default=[0.0, 1.1])
