import operator
import sys
from collections import Counter

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
    def func(f): return f.im_func

def toposort(end_nodes, get_parents=operator.attrgetter('parent_fnodes')):
    child_counts = Counter()
    stack = list(end_nodes)
    while stack:
        node = stack.pop()
        if node not in child_counts:
            stack.extend(get_parents(node))
        child_counts[node] += 1

    child_counts.subtract(end_nodes)
    childless_nodes = [n for n in end_nodes if child_counts[n] == 0]
    while childless_nodes:
        node = childless_nodes.pop()
        yield node
        for parent in get_parents(node):
            if child_counts[parent] == 1:
                childless_nodes.append(parent)
            else:
                child_counts[parent] -= 1

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
