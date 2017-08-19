"""Demonstrates how to use the tracer module, independent of autodiff, by
creating a trace that prints out functions and their arguments as they're being
evaluated"""
import autograd.numpy as np  # autograd has already wrapped numpy for us
from autograd.tracer import trace, Node

num_vars = 0  # global counter to create unique variable names
class PrintNode(Node):
    def process_local_data(self, fun, args, kwargs, ans, argnums):
        global num_vars
        self.varname = chr(num_vars + 65)
        num_vars += 1
        args_or_vars = list(args)
        for argnum, parent in zip(argnums, self.parents):
            args_or_vars[argnum] = parent.varname
        if fun:
            print '{} = {}({}) = {}'.format(
                self.varname,
                fun.__name__,
                ','.join(map(str, args_or_vars)),
                ans)
        else:
            print '{} = {}'.format(self.varname, ans)

def print_trace(f, x):
    print '----- starting trace -----'
    trace(PrintNode, f, x)
    print '----- done -----'

def avg(x, y):
    return (x + y) / 2

def fun(x):
    y = np.sin(x + x)
    return avg(y, y)

print_trace(fun, 1.23)

# Traces can be neseted, so we can also trace through grad(fun)
from autograd import grad
print_trace(grad(fun), 1.0)
