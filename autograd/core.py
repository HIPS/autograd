from .tracer import trace, Node, Box, register_box, toposort
from .vspace import vspace, assert_vspace_match, register_vspace, identity_vjp
from .misc import unary_to_nary

 # other modules expect these here but we don't actually need them
from .tracer import getval, primitive, nograd_primitive, isbox
from .vspace import VSpace, vspace_flatten

@unary_to_nary
def make_vjp(fun):
    def vjp_maker(x):
        end_value, end_node =  trace(VJPNode, fun, x)
        if end_node is None:
            def vjp(g): return vspace(x).zeros()
        else:
            def vjp(g): return backward_pass(g, end_node)
        return vjp, end_value
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
