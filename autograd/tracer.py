import warnings
from contextlib import contextmanager
from .util import subvals, wraps

def trace(node_type, fun, x):
    with trace_stack.new_trace() as t:
        start_node = node_type(t, [], None, (), {}, x, [])
        start_box = new_box(x, start_node)
        end_box = fun(start_box)
        if isbox(end_box) and end_box._trace == start_box._trace:
            return end_box._value, end_box._node
        else:
            warnings.warn("Output seems independent of input.")
            return end_box, None

class Node(object):
    def __init__(self, trace, parents, *local_data):
        self.trace = trace
        self.parents = parents
        self.process_local_data(*local_data)

    def process_local_data(fun, args, kwargs, ans, argnums): pass

def primitive(f_raw):
    """
    Wraps a function so that its gradient can be specified and its invocation
    can be recorded. For examples, see the docs."""
    @wraps(f_raw)
    def f_wrapped(*args, **kwargs):
        boxed_args, trace, node_constructor = find_top_boxed_args(args)
        if boxed_args:
            argvals = subvals(args, [(argnum, box._value) for argnum, box in boxed_args])
            parents = [box._node for _     , box in boxed_args]
            argnums = [argnum    for argnum, _   in boxed_args]
            ans = f_wrapped(*argvals, **kwargs)
            node = node_constructor(trace, parents, f_wrapped, argvals, kwargs, ans, argnums)
            return new_box(ans, node)
        else:
            return f_raw(*args, **kwargs)
    return f_wrapped

def notrace_primitive(f_raw):
    @wraps(f_raw)
    def f_wrapped(*args, **kwargs):
        argvals = map(getval, args)
        return f_raw(*argvals, **kwargs)
    f_wrapped._is_primitive = True
    return f_wrapped

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
                top_node_type = type(arg._node)
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
    __slots__ = ['_value', '_trace', '_node']
    def __init__(self, value, node):
        self._value = value
        self._node = node
        self._trace = node.trace

    def __bool__(self):
        return bool(self._value)

    __nonzero__ = __bool__

    def __str__(self):
        return "Autograd {0} with value {1}".format(
            type(self).__name__, str(self._value))

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

isbox  = lambda x: type(x) in box_types  # almost 3X faster than isinstance(x, Box)
getval = lambda x: getval(x._value) if isbox(x) else x
