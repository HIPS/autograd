from __future__ import absolute_import
import sys
import types
from functools import partial
import warnings
from .errors import defgrad_deprecated
from contextlib import contextmanager

def trace(node_type, fun, x):
    with trace_stack.new_trace() as t:
        start_node = node_type(t, [], None, (), {}, x, [])
        start_box = new_box(x, start_node)
        end_box = fun(start_box)
        if isbox(end_box) and end_box._trace == start_box._trace:
            return end_box.value, end_box.node
        else:
            warnings.warn("Output seems independent of input.")
            return end_box, None

class Node(object):
    def __init__(self, trace, parents, *local_data):
        self.trace = trace
        self.parents = parents
        self.process_local_data(*local_data)

    def process_local_data(fun, args, kwargs, ans, argnums): pass

class primitive(object):
    """
    Wraps a function so that its gradient can be specified and its invocation
    can be recorded. For examples, see the docs."""
    def __init__(self, fun):
        self.fun = fun
        self.vjps = {}
        self.__name__ = fun.__name__
        self.__doc__ = fun.__doc__

    def __call__(self, *args, **kwargs):
        boxed_args, trace, node_constructor = find_top_boxed_args(args)
        if boxed_args:
            argvals = subvals(args, [(argnum, box.value) for argnum, box in boxed_args])
            ans = self(*argvals, **kwargs)
            parents, argnums = zip(*[(box.node, argnum) for argnum, box in boxed_args])
            node = node_constructor(trace, parents, self, argvals, kwargs, ans, argnums)
            return new_box(ans, node)
        else:
            return self.fun(*args, **kwargs)

    def vjp(self, argnum, ans, vs, gvs, args, kwargs):
        try:
            return self.vjps[argnum](ans, vs, gvs, *args, **kwargs)
        except KeyError:
            if self.vjps == {}:
                errstr = "Gradient of {0} not yet implemented."
            else:
                errstr = "Gradient of {0} w.r.t. arg number {1} not yet implemented."
            raise NotImplementedError(errstr.format(self.fun.__name__, argnum))

    def defvjp(self, vjpmaker, argnum=0):
        vjpmaker.__name__ = "VJP_{}_of_{}".format(argnum, self.__name__)
        self.vjps[argnum] = vjpmaker

    def defvjps(self, vjpmaker, argnums):
        for argnum in argnums:
            self.defvjp(partial(vjpmaker, argnum), argnum)

    def defvjp_is_zero(self, argnums=(0,)):
        for argnum in argnums:
            self.vjps[argnum] = zero_vjp

    def __repr__(self):
        return self.__name__

    if sys.version_info >= (3,):
        def __get__(self, obj, objtype):
            return types.MethodType(self, obj)
    else:
        def __get__(self, obj, objtype):
            return types.MethodType(self, obj, objtype)

    def defgrad(self, gradfun, argnum=0):
        warnings.warn(defgrad_deprecated)
        def vjp(ans, vs, gvs, *args, **kwargs):
            return gradfun(ans, *args, **kwargs)
        self.defvjp(vjp, argnum)

def zero_vjp(ans, vs, gvs, *args, **kwargs):
    return lambda g: vs.zeros()

class nograd_primitive(primitive):
    def __call__(self, *args, **kwargs):
        argvals = map(getval, args)
        return self.fun(*argvals, **kwargs)

def find_top_boxed_args(args):
    top_trace = -1
    top_boxes = []
    top_node_type = None
    for argnum, arg in enumerate(args):
        if isbox(arg):
            trace = arg._trace
            if trace > top_trace:
                top_boxes = [(argnum, arg)]
                top_trace = trace
                top_node_type = type(arg.node)
            elif trace == top_trace:
                top_boxes.append((argnum, arg))
    return top_boxes, top_trace, top_node_type

class TraceStack(object):
    def __init__(self):
        self.top = -1
    @contextmanager
    def new_trace(self):
        self.top += 1
        yield self.top
        self.top -= 1
trace_stack = TraceStack()

class Box(object):
    __slots__ = ['vspace', 'value', '_trace', 'node']
    def __init__(self, value, node):
        self.value = value
        self.node = node
        self._trace = node.trace
        self.vspace = node.vspace

    def __bool__(self):
        return bool(self.value)

    __nonzero__ = __bool__

    def __str__(self):
        return "Autograd {0} with value {1}".format(
            type(self).__name__, str(self.value))

def toposort(end_node):
    child_counts = {}
    stack = [end_node]
    while stack:
        node = stack.pop()
        if node in child_counts:
            child_counts[node] += 1
        else:
            child_counts[node] = 1
            stack.extend(node.parents)

    childless_nodes = [end_node]
    while childless_nodes:
        node = childless_nodes.pop()
        yield node
        for parent in node.parents:
            if child_counts[parent] == 1:
                childless_nodes.append(parent)
            else:
                child_counts[parent] -= 1

box_type_mappings = {}
box_types = set()
def register_box(box_type, value_type):
    box_types.add(box_type)
    box_type_mappings[value_type] = box_type
    box_type_mappings[box_type] = box_type

def new_box(value, node):
    try:
        return box_type_mappings[type(value)](value, node)
    except KeyError:
        raise TypeError("Can't differentiate w.r.t. type {}".format(type(value)))

def subvals(x, ivs):
    x_ = list(x)
    for i, v in ivs:
        x_[i] = v
    return tuple(x_)

isbox = lambda x: type(x) in box_types
getval = lambda x: getval(x.value) if isbox(x) else x
