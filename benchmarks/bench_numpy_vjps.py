from autograd import make_vjp

import autograd.numpy as np
import autograd.numpy.random as npr

dot_0 = lambda a, b, g: make_vjp(np.dot, argnum=0)(a, b)[0](g)
dot_1 = lambda a, b, g: make_vjp(np.dot, argnum=1)(a, b)[0](g)

dot_0_0 = lambda a, b, g: make_vjp(dot_0, argnum=0)(a, b, g)[0](a)
dot_0_1 = lambda a, b, g: make_vjp(dot_0, argnum=1)(a, b, g)[0](a)
dot_0_2 = lambda a, b, g: make_vjp(dot_0, argnum=2)(a, b, g)[0](a)

dot_1_0 = lambda a, b, g: make_vjp(dot_1, argnum=0)(a, b, g)[0](b)
dot_1_1 = lambda a, b, g: make_vjp(dot_1, argnum=1)(a, b, g)[0](b)
dot_1_2 = lambda a, b, g: make_vjp(dot_1, argnum=2)(a, b, g)[0](b)

a = npr.randn(2, 3, 4, 5)
b = npr.randn(2, 3, 5, 4)
g = npr.randn(2, 3, 4, 2, 3, 4)

def time_dot_0():
    dot_0(a, b, g)

def time_dot_1():
    dot_1(a, b, g)

def time_dot_0_0():
    dot_0_0(a, b, g)

def time_dot_0_1():
    dot_0_1(a, b, g)

def time_dot_0_2():
    dot_0_2(a, b, g)

def time_dot_1_0():
    dot_1_0(a, b, g)

def time_dot_1_1():
    dot_1_1(a, b, g)

def time_dot_1_2():
    dot_1_2(a, b, g)

tensordot_0 = lambda A, B, G: make_vjp(np.tensordot, argnum=0)(A, B, 2)[0](G)
tensordot_1 = lambda A, B, G: make_vjp(np.tensordot, argnum=1)(A, B, 2)[0](G)

tensordot_0_0 = lambda A, B, G: make_vjp(tensordot_0, argnum=0)(A, B, G)[0](A)
tensordot_0_1 = lambda A, B, G: make_vjp(tensordot_0, argnum=1)(A, B, G)[0](A)
tensordot_0_2 = lambda A, B, G: make_vjp(tensordot_0, argnum=2)(A, B, G)[0](A)

tensordot_1_0 = lambda A, B, G: make_vjp(tensordot_1, argnum=0)(A, B, G)[0](B)
tensordot_1_1 = lambda A, B, G: make_vjp(tensordot_1, argnum=1)(A, B, G)[0](B)
tensordot_1_2 = lambda A, B, G: make_vjp(tensordot_1, argnum=2)(A, B, G)[0](B)

A = npr.randn(2, 3, 5, 4)
B = npr.randn(5, 4, 2, 3)
G = npr.randn(2, 3, 2, 3)

def time_tensordot_0():
    tensordot_0(A, B, G)

def time_tensordot_1():
    tensordot_1(A, B, G)

def time_tensordot_0_0():
    tensordot_0_0(A, B, G)

def time_tensordot_0_1():
    tensordot_0_1(A, B, G)

def time_tensordot_0_2():
    tensordot_0_2(A, B, G)

def time_tensordot_1_0():
    tensordot_1_0(A, B, G)

def time_tensordot_1_1():
    tensordot_1_1(A, B, G)

def time_tensordot_1_2():
    tensordot_1_2(A, B, G)

