import numpy as onp
import autograd.numpy as np
from autograd import grad
try:
    from autograd.core import vspace, VJPNode, backward_pass
    from autograd.tracer import trace, new_box
    MASTER_BRANCH = False
except ImportError:
    from autograd.core import (vspace, forward_pass, backward_pass,
                               new_progenitor) 
    MASTER_BRANCH = True

## SHORT FUNCTION
def f_short(x):
    return x**2

def time_short_fun():
    f_short(2.)

def time_short_forward_pass():
    if MASTER_BRANCH:
        forward_pass(f_short, (2.,), {})
    else:
        start_node = VJPNode.new_root()
        trace(start_node, f_short, x)

def time_short_backward_pass():
    if MASTER_BRANCH:
        backward_pass(1., short_end_node, short_start_node)
    else:
        backward_pass(1., short_end_node)

def time_short_grad():
    grad(f_short)(2.)

## LONG FUNCTION
def f_long(x):
    for i in range(50):
        x = np.sin(x)
    return x

def time_long_fun():
    f_long(2.)

def time_long_forward_pass():
    if MASTER_BRANCH:
        forward_pass(f_long, (2.,), {})
    else:
        start_node = VJPNode.new_root()
        trace(start_node, f_long, x)

def time_long_backward_pass():
    if MASTER_BRANCH:
        backward_pass(1., long_end_node, long_start_node)
    else:
        backward_pass(1., long_end_node)

def time_long_grad():
    grad(f_long)(2.)

## 'PEARLMUTTER TEST' FUNCTION
def fan_out_fan_in(x):
    for i in range(10**4):
        x = (x + x)/2.0
    return np.sum(x)

def time_fan_out_fan_in_fun():
    fan_out_fan_in(2.)

def time_fan_out_fan_in_forward_pass():
    if MASTER_BRANCH:
        forward_pass(fan_out_fan_in, (2.,), {})
    else:
        start_node = VJPNode.new_root()
        trace(start_node, fan_out_fan_in, x)

def time_fan_out_fan_in_backward_pass():
    if MASTER_BRANCH:
        backward_pass(1., fan_end_node, fan_start_node)
    else:
        backward_pass(1., fan_end_node)

def time_fan_out_fan_in_grad():
    grad(fan_out_fan_in)(2.)

## UNIT BENCHMARKS
def time_vspace_float():
    vspace(1.)

A = np.array([[1., 2., 3.]])

def time_vspace_array():
    vspace(A)

def time_new_box_float():
    new_box(1., 0, start_node)

def time_new_box_array():
    new_box(A, 0, start_node)

def time_exp_call():
    onp.exp(2.)

def time_exp_primitive_call_unboxed():
    np.exp(2.)

def time_exp_primitive_call_boxed():
    if MASTER_BRANCH:
        np.exp(progenitor)
    else:
        np.exp(start_box)

def time_no_autograd_control():
    # Test whether the benchmarking machine is running slowly independent of autograd
    A = np.random.randn(200, 200)
    np.dot(A, A)

if MASTER_BRANCH:
    short_start_node, short_end_node = forward_pass(f_short, (2.,), {})
    long_start_node, long_end_node = forward_pass(f_long, (2.,), {})
    fan_start_node, fan_end_node = forward_pass(fan_out_fan_in, (2.,), {})
    progenitor = new_progenitor(2.)
else:
    x = 2.
    start_node = VJPNode.new_root()
    start_box = new_box(x, 0, start_node)
    _, short_end_node = trace(VJPNode.new_root(), f_short, x)
    _, long_end_node  = trace(VJPNode.new_root(), f_long, x)
    _, fan_end_node   = trace(VJPNode.new_root(), fan_out_fan_in, x)
