from autograd import make_vjp

import autograd.numpy as np
import autograd.numpy.random as npr

dot_0 = lambda A, B, g: make_vjp(np.dot)(A, B)[0](g)
dot_1 = lambda A, B, g: make_vjp(np.dot, argnum=1)(A, B)[0](g)

dot_0_0 = lambda A, B, g: make_vjp(dot_0)(A, B, g)[0](A)
dot_0_1 = lambda A, B, g: make_vjp(dot_0)(A, B, g)[0](A)
dot_0_2 = lambda A, B, g: make_vjp(dot_0)(A, B, g)[0](A)

dot_1_0 = lambda A, B, g: make_vjp(dot_1)(A, B, g)[0](B)
dot_1_1 = lambda A, B, g: make_vjp(dot_1)(A, B, g)[0](B)
dot_1_2 = lambda A, B, g: make_vjp(dot_1)(A, B, g)[0](B)

A = npr.randn(2, 3, 4, 5)
B = npr.randn(2, 3, 5, 4)
g = npr.randn(2, 3, 4, 2, 3, 4)

def time_dot_0():
    dot_0(A, B, g)

def time_dot_1():
    dot_1(A, B, g)

def time_dot_0_0():
    dot_0_0(A, B, g)

def time_dot_0_1():
    dot_0_1(A, B, g)

def time_dot_0_2():
    dot_0_2(A, B, g)

def time_dot_1_0():
    dot_1_0(A, B, g)

def time_dot_1_1():
    dot_1_1(A, B, g)

def time_dot_1_2():
    dot_1_2(A, B, g)
