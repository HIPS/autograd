import threading
from functools import partial
from itertools import repeat

from autograd.tracer import Node, trace
from autograd.util import subvals, toposort
from autograd.wrap_util import wraps


class ConstGraphNode(Node):
    __slots__ = ["parents", "partial_fun"]

    def __init__(self, value, fun, args, kwargs, parent_argnums, parents):
        args = subvals(args, zip(parent_argnums, repeat(None)))

        def partial_fun(partial_args):
            return fun(*subvals(args, zip(parent_argnums, partial_args)), **kwargs)

        self.parents = parents
        self.partial_fun = partial_fun

    def initialize_root(self):
        self.parents = []


def const_graph_unary(fun):
    graph = []
    _fun = [fun]  # Allow fun to be freed, since it may have bound args

    # See https://py-free-threading.github.io/porting/#locking
    # The graph is traced once and cached for reuse across calls. The cache is
    # shared. We guard the one-time fill of the cache with a lock so that we
    # keep concurrent first calls thread-safe. Without it, two threads
    # could both see the graph as empty and then trace, and both would try to
    # call _fun.pop(), which would raise an IndexError. The lock is only held
    # during the one-time fill of the cache, so subsequent calls to the cached
    # graph are not blocked by the lock.
    lock = threading.Lock()

    def maybe_cached_fun(x):
        if not graph:
            with lock:
                if not graph:  # another thread may have filled it while we waited
                    start_node = ConstGraphNode.new_root()
                    end_value, end_node = trace(start_node, _fun.pop(), x)
                    if end_node is None:
                        raise Exception("Output is independent of input")
                    graph.append(list(toposort(end_node))[::-1])
                    return end_value
        _graph = graph[0]
        vals = {_graph[0]: x}
        for node in _graph[1:]:
            vals[node] = node.partial_fun([vals[p] for p in node.parents])
        return vals[node]

    return maybe_cached_fun


def const_graph(fun, *args, **kwargs):
    partial_fun = partial(fun, *args, **kwargs)
    unary_fun = lambda args: partial_fun(*args)
    maybe_cached_unary_fun = const_graph_unary(unary_fun)

    @wraps(fun)
    def _fun(*args):
        return maybe_cached_unary_fun(args)

    return _fun


class FullGraphNode(Node):
    __slots__ = ["value", "recipe"]

    def __init__(self, value, fun, args, kwargs, parent_argnums, parents):
        self.value = value
        self.recipe = (fun, args, kwargs, zip(parent_argnums, parents))

    def initialize_root(self):
        self.value = None
        self.recipe = (lambda x: x, (), {}, [])


def full_graph(fun, *args, **kwargs):
    unary_fun = lambda args: fun(*args, **kwargs)
    start_node = FullGraphNode.new_root()
    end_value, end_node = trace(start_node, unary_fun, args)
    return end_node
