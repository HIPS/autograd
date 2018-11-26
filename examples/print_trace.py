"""Demonstrates how to use the tracer module, independent of autodiff, by
creating a trace that prints out functions and their arguments as they're being
evaluated"""
from __future__ import print_function
import autograd.numpy as np  # autograd has already wrapped numpy for us
from autograd.tracer import trace, Node

class PrintNode(Node):
    def __init__(self, value, fun, args, kwargs, parent_argnums, parents):
        self.varname_generator = parents[0].varname_generator
        self.varname = next(self.varname_generator)
        args_or_vars = list(args)
        for argnum, parent in zip(parent_argnums, parents):
            args_or_vars[argnum] = parent.varname
        print('{} = {}({}) = {}'.format(
            self.varname, fun.__name__, ','.join(map(str, args_or_vars)), value))

    def initialize_root(self, x):
        self.varname_generator = make_varname_generator()
        self.varname = next(self.varname_generator)
        print('{} = {}'.format(self.varname, x))

def make_varname_generator():
    for i in range(65, 91):
        yield chr(i)
    raise Exception("Ran out of alphabet!")

def print_trace(f, x):
    start_node = PrintNode.new_root(x)
    trace(start_node, f, x)
    print()

def avg(x, y):
    return (x + y) / 2
def fun(x):
    y = np.sin(x + x)
    return avg(y, y)

print_trace(fun, 1.23)

# Traces can be nested, so we can also trace through grad(fun)
from autograd import grad
print_trace(grad(fun), 1.0)
