import autograd.core as core
from autograd import grad
import autograd.numpy as np
import numpy as onp

## SHORT FUNCTION
def f_short(x):
    return x**2

def time_short_forward_pass():
    core.forward_pass(f_short, (2.,), {})

short_start_node, short_end_node = core.forward_pass(f_short, (2.,), {})

def time_short_backward_pass():
    core.backward_pass(1., short_end_node.node, short_start_node)

def time_short_grad():
    grad(f_short)(2.)

## LONG FUNCTION
def f_long(x):
    for i in range(50):
        x = np.sin(x)
    return x

def time_long_forward_pass():
    core.forward_pass(f_long, (2.,), {})

long_start_node, long_end_node = core.forward_pass(f_long, (2.,), {})

def time_long_backward_pass():
    core.backward_pass(1., long_end_node.node, long_start_node)

def time_long_grad():
    grad(f_long)(2.)

## 'PEARLMUTTER TEST' FUNCTION
def fan_out_fan_in(x):
    for i in range(10**4):
        x = (x + x)/2.0
    return np.sum(x)

def time_fan_out_fan_in_forward_pass():
    core.forward_pass(fan_out_fan_in, (2.,), {})

fan_start_node, fan_end_node = core.forward_pass(fan_out_fan_in, (2.,), {})

def time_fan_out_fan_in_backward_pass():
    core.backward_pass(1., fan_end_node.node, fan_start_node)

def time_fan_out_fan_in_grad():
    grad(fan_out_fan_in)(2.)

## UNIT BENCHMARKS
def time_vspace_float():
    core.vspace(1.)

A = np.array([[1., 2., 3.]])

def time_vspace_array():
    core.vspace(A)

progenitors = {'dummy'}

def time_new_box_float():
    core.new_box(1., progenitors)

def time_new_box_array():
    core.new_box(A, progenitors)

def time_exp_call():
    onp.exp(2.)

def time_exp_primitive_call_unboxed():
    np.exp(2.)

progenitor = core.new_progenitor(2.)
def time_exp_primitive_call_boxed():
    np.exp(progenitor)
