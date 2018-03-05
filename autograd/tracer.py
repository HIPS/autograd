import warnings
from contextlib import contextmanager
from collections import defaultdict
from functools import partial
from itertools import count
from operator import attrgetter
from .fmap_util import limited_fmap, apply
from .wrap_util import wraps

def trace(start_nodes, fun, xs, fmap_in, fmap_out):
    with trace_stack.new_trace() as t:
        start_boxes = fmap_in(partial(new_box, t), xs, start_nodes)
        end_boxes = fun(start_boxes)
        return unpack_boxes(fmap_out, end_boxes, t)

def unpack_boxes(fmap, boxes, trace):
    is_top_box = lambda box: isbox(box) and box._trace == trace
    valid_boxes = fmap(is_top_box, boxes)
    nodes  = fmap(lambda cond, b: b._node if cond else None, valid_boxes, boxes)
    l_fmap = limited_fmap(fmap, valid_boxes)
    values = l_fmap(attrgetter('_value'), boxes)
    return values, nodes, l_fmap

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
    return general_primitive(f_raw, map, apply)

def general_primitive(f_raw, fmap_in, fmap_out):
    """
    Wraps a function so that its gradient can be specified and its invocation
    can be recorded. For examples, see the docs."""
    @wraps(f_raw)
    def f_wrapped(*args, **kwargs):
        boxed_args = []
        fmap_in(lambda arg: isbox(arg) and boxed_args.append(arg), args)
        if boxed_args:
            top_box = max(boxed_args, key=lambda box: box._trace)
            argvals, parents, parent_fmap = unpack_boxes(
                fmap_in, args, top_box._trace)
            if f_wrapped in notrace_primitives[type(top_box._node)]:
                return f_wrapped(*argvals, **kwargs)
            ans = f_wrapped(*argvals, **kwargs)
            output_nodes, lfmap_out = top_box._node.make_nodes(
                ans, f_wrapped, argvals, kwargs, parents, parent_fmap)
            return lfmap_out(partial(new_box, top_box._trace), ans, output_nodes)
        else:
            return f_raw(*args, **kwargs)

    f_wrapped.fun = f_raw
    f_wrapped._is_autograd_primitive = True
    f_wrapped._fmap = fmap_in
    f_wrapped._fmap_out = fmap_out
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
def new_box(trace, value, node):
    try:
        return box_type_mappings[type(value)](value, trace, node)
    except KeyError:
        raise TypeError("Can't differentiate w.r.t. type {}".format(type(value)))

box_types = Box.types
isbox  = lambda x: type(x) in box_types  # almost 3X faster than isinstance(x, Box)
getval = lambda x: getval(x._value) if isbox(x) else x
