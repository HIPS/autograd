import warnings
from collections import defaultdict
from contextlib import contextmanager

import numpy as np

import autograd

from .util import subvals, toposort
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


class Node:
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


trace_primitives_map = {}


def primitive(f_raw):
    """
    Wraps a function so that its gradient can be specified and its invocation
    can be recorded. For examples, see the docs."""

    @wraps(f_raw)
    def f_wrapped(*args, called_by_autograd_dispatcher=False, **kwargs):
        boxed_args, trace, node_constructor = find_top_boxed_args(args)
        if boxed_args:
            # If we are a wrapper around a ufunc, first forward further handling to
            # the ufunc dispatching mechanism (if we aren't already running inside it)
            # by calling the ufunc. This allows other operands to also try to handle
            # the call (it's still possible our handling attempt below will get the
            # first shot; the handlers order is determined by the dispatch mechanism).
            #
            # For example, consider multiplying an ndarray wrapped inside an ArrayBox
            # by an xarray.DataArray. The handling below will fail: The ndarray will
            # be unboxed and multiplied by the DataArray resulting in a DataArray,
            # for which `new_box` will raise an exception. In contrast, the DataArray's
            # handling of the call might succeed: it might contain an ndarray, either
            # plain or boxed in an ArrayBox, in which case it will be multiplied by
            # the other ArrayBox yielding a new ArrayBox, which will be stored in a new
            # DataArray.
            if (
                isinstance(f_raw, np.ufunc)
                and not called_by_autograd_dispatcher
                and any(isinstance(arg, autograd.numpy.numpy_boxes.ArrayBox) for arg in args)
            ):
                return f_raw(*args, **kwargs)

            argvals = subvals(args, [(argnum, box._value) for argnum, box in boxed_args])
            if f_wrapped in notrace_primitives[node_constructor]:
                return f_wrapped(
                    *argvals, called_by_autograd_dispatcher=called_by_autograd_dispatcher, **kwargs
                )
            parents = tuple(box._node for _, box in boxed_args)
            argnums = tuple(argnum for argnum, _ in boxed_args)
            ans = f_wrapped(*argvals, called_by_autograd_dispatcher=called_by_autograd_dispatcher, **kwargs)
            node = node_constructor(ans, f_wrapped, argvals, kwargs, argnums, parents)
            try:
                box = new_box(ans, trace, node)
                return box
            except:
                if called_by_autograd_dispatcher:
                    raise NotImplementedError
                raise
        else:
            return f_raw(*args, **kwargs)

    f_wrapped.fun = f_raw
    f_wrapped._is_autograd_primitive = True
    trace_primitives_map[f_raw] = f_wrapped
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


class TraceStack:
    def __init__(self):
        self.top = -1

    @contextmanager
    def new_trace(self):
        self.top += 1
        yield self.top
        self.top -= 1


trace_stack = TraceStack()


class Box:
    type_mappings = {}
    types = set()

    __slots__ = ["_value", "_trace", "_node"]

    def __init__(self, value, trace, node):
        self._value = value
        self._node = node
        self._trace = trace

    def __bool__(self):
        return bool(self._value)

    __nonzero__ = __bool__

    def __str__(self):
        return f"Autograd {type(self).__name__} with value {str(self._value)}"

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
        raise TypeError(f"Can't differentiate w.r.t. type {type(value)}")


box_types = Box.types
isbox = lambda x: type(x) in box_types  # almost 3X faster than isinstance(x, Box)
getval = lambda x: getval(x._value) if isbox(x) else x
