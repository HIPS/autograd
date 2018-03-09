from itertools import repeat
from autograd.wrap_util import wraps
from autograd.tracer import trace, Node
from autograd.util import toposort
from autograd.fmap_util import apply, limited_fmap
from functools import partial

class ConstGraphNode(Node):
    __slots__ = ['parents', 'parent_fmap', 'partial_fun']
    def __init__(self, parents, parent_fmap, partial_fun):
        self.parents = parents
        self.parent_fmap = parent_fmap
        self.partial_fun = partial_fun

    def initialize_root(self, _):
        self.parents = ()
        self.parent_fmap = lambda *args: ()

    def process_primitive(self, ans, fun, args, kwargs, parents):
        assert fun.fmap_out is apply  # only works for single-output primitives
        parent_fmap = limited_fmap(fun.fmap_in, parents)
        static_args = parent_fmap(lambda _: None, args)
        def partial_fun(dynamic_args):
            complete_args = parent_fmap(
                lambda _, dynamic_arg: dynamic_arg, static_args, dynamic_args)
            return fun(*complete_args)
        return ConstGraphNode(parents, parent_fmap, partial_fun)

def const_graph(fun, *args, **kwargs):
    fun = [partial(fun, *args, **kwargs)] # Allow fun to be freed, since it may have bound args
    cache = {}
    def maybe_cached_fun(*xs):
        if cache:
            graph = cache['graph']
            start_nodes = cache['start_nodes']
            vals = {n: x for n, x in zip(start_nodes, xs)}
            for node in graph:
                if node in start_nodes: continue
                vals[node] = node.partial_fun(node.parent_fmap(vals.get, node.parents))
            return vals[node]
        else:
            start_nodes = map(ConstGraphNode.new_root, xs)
            end_value, end_node = trace(start_nodes, fun.pop(), xs, map, apply)
            if end_node is None:
                raise Exception("Output is independent of input")
            graph = list(toposort(
                [end_node], lambda n: filter(bool, n.parents)))[::-1]
            cache['graph'] = graph
            cache['start_nodes'] = start_nodes
            return end_value
    return wraps(fun)(maybe_cached_fun)

# TODO: update this to new tracer interface
class FullGraphNode(Node):
    __slots__ = ['value', 'recipe']
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
