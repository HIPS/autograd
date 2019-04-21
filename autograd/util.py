import operator
import sys

def subvals(x, ivs):
    x_ = list(x)
    for i, v in ivs:
        x_[i] = v
    return tuple(x_)

def subval(x, i, v):
    x_ = list(x)
    x_[i] = v
    return tuple(x_)

if sys.version_info >= (3,):
    def func(f): return f
else:
    def func(f): return f.__func__

def toposort(end_node, parents=operator.attrgetter('parents')):
    child_counts = {}
    stack = [end_node]
    while stack:
        node = stack.pop()
        if node in child_counts:
            child_counts[node] += 1
        else:
            child_counts[node] = 1
            stack.extend(parents(node))

    childless_nodes = [end_node]
    while childless_nodes:
        node = childless_nodes.pop()
        yield node
        for parent in parents(node):
            if child_counts[parent] == 1:
                childless_nodes.append(parent)
            else:
                child_counts[parent] -= 1


def typeof(x):
    """
    A Modified type function that returns np.ndarray for any array-like

    This improves portability of autograd to other projects that might support
    the numpy API, despite not being exactly numpy.
    """
    if all(hasattr(x, attr) for attr in ['__array_ufunc__', 'shape', 'dtype']):
        import numpy
        return numpy.ndarray
    else:
        return type(x)

# -------------------- deprecation warnings -----------------------

import warnings
deprecation_msg = """
The quick_grad_check function is deprecated. See the update guide:
https://github.com/HIPS/autograd/blob/master/docs/updateguide.md"""

def quick_grad_check(fun, arg0, extra_args=(), kwargs={}, verbose=True,
                     eps=1e-4, rtol=1e-4, atol=1e-6, rs=None):
    warnings.warn(deprecation_msg)
    from autograd.test_util import check_grads
    fun_ = lambda arg0: fun(arg0, *extra_args, **kwargs)
    check_grads(fun_, modes=['rev'], order=1)(arg0)
