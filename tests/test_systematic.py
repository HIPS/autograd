import autograd.numpy.random as npr
import autograd.numpy as np
import operator as op
from numpy_utils import combo_check, stat_check, unary_ufunc_check, binary_ufunc_check
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

# Unary ufunc tests

def test_sin():     unary_ufunc_check(np.sin) 
def test_abs():     unary_ufunc_check(np.abs)
def test_arccosh(): unary_ufunc_check(np.arccosh, lims=[1.1, 4.0])
def test_arcsinh(): unary_ufunc_check(np.arcsinh, lims=[-0.9, 0.9])
def test_arctanh(): unary_ufunc_check(np.arctanh, lims=[-0.9, 0.9])
def test_ceil():    unary_ufunc_check(np.ceil, lims=[-1.5, 1.5])
def test_cos():     unary_ufunc_check(np.cos)
def test_cosh():    unary_ufunc_check(np.cosh)
def test_deg2rad(): unary_ufunc_check(np.deg2rad)
def test_exp():     unary_ufunc_check(np.exp)
def test_exp2():    unary_ufunc_check(np.exp2)
def test_expm1():   unary_ufunc_check(np.expm1)
def test_fabs():    unary_ufunc_check(np.fabs)
def test_floor():   unary_ufunc_check(np.floor, lims=[-1.5, 1.5])
def test_log():     unary_ufunc_check(np.log,   lims=[0.2, 2.0])
def test_log10():   unary_ufunc_check(np.log10, lims=[0.2, 2.0])
def test_log1p():   unary_ufunc_check(np.log1p, lims=[0.2, 2.0])
def test_log2():    unary_ufunc_check(np.log2,  lims=[0.2, 2.0])
def test_rad2deg(): unary_ufunc_check(lambda x : np.rad2deg(x)/50.0)
def test_sign():    unary_ufunc_check(np.sign)
def test_sin():     unary_ufunc_check(np.sin)
def test_sinh():    unary_ufunc_check(np.sinh)
def test_sqrt():    unary_ufunc_check(np.sqrt, lims=[1.0, 3.0])
def test_square():  unary_ufunc_check(np.square)
def test_tan():     unary_ufunc_check(np.tan, lims=[-1.1, 1.1])
def test_tanh():    unary_ufunc_check(np.tanh)

# Binary ufunc tests

def test_add(): binary_ufunc_check(np.add)
def test_logaddexp(): binary_ufunc_check(np.logaddexp)
def test_logaddexp2(): binary_ufunc_check(np.logaddexp2)

def test_op_mul(): binary_ufunc_check(op.mul)
def test_op_add(): binary_ufunc_check(op.add)
def test_op_sub(): binary_ufunc_check(op.sub)
def test_op_div(): binary_ufunc_check(op.div, lims_B=[0.3, 2.0])
def test_op_pow(): binary_ufunc_check(op.pow, lims_A=[0.5, 2.0])

# Misc tests

def test_transpose(): combo_check(np.transpose, [0],
                                  [npr.randn(2, 3, 4)],
                                  axes = [None, [0, 1, 2], [0, 2, 1],
                                                [2, 0, 1], [2, 1, 0],
                                                [1, 0, 2], [1, 2, 0]])

R = npr.randn
def test_dot(): combo_check(np.dot, [0, 1],
                            [1.5, R(3), R(2, 3)],
                            [0.3, R(3), R(3, 4)])


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

# Need custom tests because gradient is undefined when arguments are identical.
def test_maximum(): combo_check(np.maximum, [0, 1],
                               [R(1), R(1,4), R(3, 4)],
                               [R(1), R(1,4), R(3, 4)])

def test_minimum(): combo_check(np.minimum, [0, 1],
                               [R(1), R(1,4), R(3, 4)],
                               [R(1), R(1,4), R(3, 4)])
