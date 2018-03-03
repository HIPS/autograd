import warnings
from contextlib import contextmanager
from collections import defaultdict
from functools import partial
from itertools import count
from operator import attrgetter

from .wrap_util import wraps

def trace(start_node, fun, x):
    with trace_stack.new_trace() as t:
        start_box = new_box(x, t, start_node)
        end_box = fun(start_box)
        if isbox(end_box) and end_box._trace == start_box._trace:
            return end_box._value, end_box._node
        else:
            warnings.warn("Output seems independent of input.")
            return end_box, None

class Node(object):
    __slots__ = []
    def __init__(self, value, fun, args, kwargs, parent_argnums, parents):
        assert False

    def initialize_root(self, *args, **kwargs):
        assert False

    @classmethod
    def new_root(cls, *args, **kwargs):
        root = cls.__new__(cls)
        root.initialize_root(*args, **kwargs)
        return root

def primitive(f_raw):
    return general_primitive(f_raw, map)

def general_primitive(f_raw, fmap):
    """
    Wraps a function so that its gradient can be specified and its invocation
    can be recorded. For examples, see the docs."""
    @wraps(f_raw)
    def f_wrapped(*args, **kwargs):
        boxed_args = []
        fmap(lambda arg: isbox(arg) and boxed_args.append(arg), args)
        if boxed_args:
            top_box = max(boxed_args, key=lambda box: box._trace)
            def get_parent(arg):
                if isbox(arg) and arg._trace == top_box._trace:
                    return arg._node
            parents = fmap(get_parent, args)
            def parent_fmap(f, *args):
                return fmap(lambda p, x, *rest:
                            f(x, *rest) if p else x,parents, *args)
            argvals = parent_fmap(attrgetter('_value'), args)
            node_constructor = type(top_box._node)
            if f_wrapped in notrace_primitives[node_constructor]:
                return f_wrapped(*argvals, **kwargs)
            ans = f_wrapped(*argvals, **kwargs)
            node = node_constructor(ans, f_wrapped, argvals, kwargs, parents, parent_fmap)
            return new_box(ans, top_box._trace, node)
        else:
            return f_raw(*args, **kwargs)

    f_wrapped.fun = f_raw
    f_wrapped._is_autograd_primitive = True
    f_wrapped._fmap = fmap
    return f_wrapped

notrace_primitives = defaultdict(set)
def register_notrace(trace_type, primitive_fun):
    notrace_primitives[trace_type].add(primitive_fun)

def notrace_primitive(f_raw):
    @wraps(f_raw)
    def f_wrapped(*args, **kwargs):
        argvals = map(getval, args)
        return f_raw(*argvals, **kwargs)
    f_wrapped._is_primitive = True
    return f_wrapped

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
    type_mappings = {}
    types = set()

    __slots__ = ['_value', '_trace', '_node']
    def __init__(self, value, trace, node):
        self._value = value
        self._node = node
        self._trace = trace

    def __bool__(self):
        return bool(self._value)

    __nonzero__ = __bool__

    def __str__(self):
        return "Autograd {0} with value {1}".format(
            type(self).__name__, str(self._value))

    @classmethod
    def register(cls, value_type):
        Box.types.add(cls)
        Box.type_mappings[value_type] = cls
        Box.type_mappings[cls] = cls

box_type_mappings = Box.type_mappings
def new_box(value, trace, node):
    try:
        return box_type_mappings[type(value)](value, trace, node)
    except KeyError:
        raise TypeError("Can't differentiate w.r.t. type {}".format(type(value)))

box_types = Box.types
isbox  = lambda x: type(x) in box_types  # almost 3X faster than isinstance(x, Box)
getval = lambda x: getval(x._value) if isbox(x) else x

def toposort(end_node):
    child_counts = {}
    stack = [end_node]
    while stack:
        node = stack.pop()
        if node in child_counts:
            child_counts[node] += 1
        else:
            child_counts[node] = 1
            node.parent_fmap(stack.append, node.parents)

    childless_nodes = [end_node]
    while childless_nodes:
        node = childless_nodes.pop()
        yield node
        def process_parents(parent):
            if child_counts[parent] == 1:
                childless_nodes.append(parent)
            else:
                child_counts[parent] -= 1
        node.parent_fmap(process_parents, node.parents)
