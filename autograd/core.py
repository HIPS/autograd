import warnings
from .tracer import trace, Node, Box, isbox, register_box, toposort
from .vspace import vspace, assert_vspace_match, register_vspace, identity_vjp

from .tracer import getval, primitive, nograd_primitive # other modules expect these here
from .vspace import VSpace, vspace_flatten              # but we don't actually need them

def make_vjp(fun, argnum=0):
    def vjp_maker(*args, **kwargs):
        start_box, end_box = trace(VJPNode, fun, args, kwargs, argnum)
        if not isbox(end_box) or start_box._trace != end_box._trace:
            warnings.warn("Output seems independent of input.")
            def vjp(g): return start_box.vspace.zeros()
            return vjp, end_box
        else:
            def vjp(g): return backward_pass(g, end_box.node)
            return vjp, end_box.value
    return vjp_maker

def backward_pass(g, end_node):
    outgrads = {end_node : (g, False)}
    assert_vspace_match(outgrads[end_node][0], end_node.vspace)
    for node in toposort(end_node):
        if node not in outgrads: continue
        cur_outgrad = outgrads.pop(node)
        for parent, vjp in zip(node.parents, node.vjps):
            outgrad = vjp(cur_outgrad[0])
            assert_vspace_match(outgrad, parent.vspace)
            outgrads[parent] = add_outgrads(parent.vspace, outgrads.get(parent), outgrad)
    return cur_outgrad[0]

class VJPNode(Node):
    def process_local_data(self, fun, args, kwargs, ans, argnums):
        self.vspace = vspace(ans)
        self.vjps = [
            fun.vjp(argnum, ans, vspace(args[argnum]), vspace(ans), args, kwargs)
            for argnum in argnums]

def add_outgrads(vs, prev_g_flagged, g):
    sparse = type(getval(g)) == SparseObject
    if prev_g_flagged:
        prev_g, mutable = prev_g_flagged
        if mutable:
            if sparse:
                return sparse_add(prev_g, g), True
            else:
                return vs.mut_add(prev_g, g), True
        else:
            if sparse:
                prev_g_mutable = vs.mut_add(vs.zeros(), prev_g)
                return sparse_add(prev_g_mutable, g), True
            else:
                return vs.add(prev_g, g), True
    else:
        if sparse:
            return sparse_add(vs.zeros(), g), True
        else:
            return g, False

@primitive
def sparse_add(x_prev, x_new): return x_new.mut_add(x_prev)
sparse_add.defvjps(identity_vjp, argnums=[0, 1])

class SparseObject(object):
    __slots__ = ['vs', 'mut_add']
    def __init__(self, vs, mut_add):
        self.vs = vs
        self.mut_add = mut_add
register_vspace(lambda x : x.vs, SparseObject)
register_box(Box, SparseObject)
