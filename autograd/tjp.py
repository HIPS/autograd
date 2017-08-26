from collections import defaultdict
from .tracer import (trace, primitive, Node, toposort)
from .vspace import vspace, assert_vspace_match, register_vspace, VSpace
from .util import unary_to_nary, func, subval
from .core import add_outgrads

def make_tjp(fun, x):
    start_node = TJPNode.new_root(x)
    end_value, end_node = trace(start_node, fun, x)
    if end_node is None:
        in_vs, out_vs = start_node.vspace, vspace(end_value)
        def tjp(G): return vspace(G)._contract(end_vs)._product(in_vs).zeros()
    else:
        def tjp(G): return tjp_backward_pass(G, end_node)
    return tjp, end_value

def tjp_backward_pass(G, end_node):
    assert_vspace_compatible(G, end_node.vspace)
    outgrads = {end_node : (G, False)}
    for node in toposort(end_node):
        cur_outgrad = outgrad.pop(node)
        for parent, tjp in node.parents_and_tjps:
            outgrad = tjp(cur_outgrad[0])
            assert_vspace_compatible(outgrad, parent.vspace)
            outgrads[parent] = add_outgrads(vspace(outgrad), outgrads.get(parent), outgrad)
    return cur_outgrad[0]

class TJPNode(Node):
    __slots__ = ['vspace', 'parents', 'parents_and_tjps']
    def __init__(self, value, fun, args, kwargs, parent_argnums, parents):
        self.vspace = vspace(value)
        self.parents = parents
        self.parents_and_tjps = [
            (parent, primitive_tjp(fun, argnum, value, parent.vspace,
                                   self.vspace, args, kwargs))
            for argnum, parent in zip(parent_argnums, parents)]

    def initialize_root(self, value):
        self.vspace = vspace(value)
        self.parents = []
        self.parents_and_tjps = []


primitive_tjps = defaultdict(dict)

def primitive_tjp(fun, argnum, ans, in_vs, out_vs, args, kwargs):
    return primitive_tjps[fun][argnum](ans, in_vs, out_vs, *args, **kwargs)

def deftjp(fun, tjpmaker, argnum=0):
    def tjp_fixed_args(ans, vs, gvs, args, kwargs):
        return tjpmaker(ans, vs, gvs, *args, **kwargs)
    primitive_tjps[fun][argnum] = tjp_fixed_args

def deftjps(fun, tjpmaker, argnums):
    for argnum in argnums:
        deftjp(fun, partial(tjpmaker, argnum), argnum)

def assert_vspace_compatible(x, vs):
    assert vspace(x).shape[-vs.ndim:] == vs.shape
