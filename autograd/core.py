from __future__ import absolute_import
import sys
import types
import numpy as np
import numpy.random as npr
from functools import partial
import warnings
from .errors import defgrad_deprecated
from contextlib import contextmanager

def make_vjp(fun, argnum=0):
    def vjp_maker(*args, **kwargs):
        start_box, end_box = forward_pass(fun, args, kwargs, argnum)
        if not isbox(end_box) or start_box._trace != end_box._trace:
            warnings.warn("Output seems independent of input.")
            def vjp(g): return start_box.vspace.zeros()
            return vjp, end_box
        else:
            def vjp(g): return backward_pass(g, end_box.node)
            return vjp, end_box.value
    return vjp_maker

def forward_pass(fun, args, kwargs, argnum=0):
    args = list(args)
    with trace_stack.new_trace() as t:
        start_node = VJPNode(None, None, None, args[argnum], [], t)
        start_box = new_box(args[argnum], start_node)
        args[argnum] = start_box
        end_box = fun(*args, **kwargs)
    return start_box, end_box

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

class Node(object): pass

class VJPNode(Node):
    def __init__(self, fun, args, kwargs, ans, numbered_parents, trace):
        self.vspace = vspace(ans)
        self.parents = [p for _, p in numbered_parents]
        self.trace = trace
        self.vjps = [fun.vjp(argnum, ans, parent.vspace, vspace(ans), args, kwargs)
                     for argnum, parent in numbered_parents]

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
            numbered_parents = [(argnum, box.node) for argnum, box in boxed_args]
            ans = self(*argvals, **kwargs)
            node = node_constructor(self, argvals, kwargs, ans, numbered_parents, trace)
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

class nograd_primitive(primitive):
    def __call__(self, *args, **kwargs):
        argvals = map(getval, args)
        return self.fun(*argvals, **kwargs)

def zero_vjp(ans, vs, gvs, *args, **kwargs):
    return lambda g: vs.zeros()

@primitive
def identity(x) : return x
identity_vjp = lambda *args: lambda g: g
identity.defvjp(identity_vjp)

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

@primitive
def vs_add(vs, x_prev, x_new): return vs._add(x_prev, x_new)
vs_add.defvjps(identity_vjp, argnums=[1,2])

@primitive
def vs_mut_add(vs, x_prev, x_new): return vs._mut_add(x_prev, x_new)
vs_mut_add.defvjps(identity_vjp, argnums=[1,2])

@primitive
def vs_covector(vs, x): return vs._covector(x)
vs_covector.defvjp(lambda ans, vs, gvs, vs_, x: lambda g:
                   gvs.covector(g), argnum=1)

@primitive
def vs_scalar_mul(vs, x, a):
    return vs._scalar_mul(x, a)
vs_scalar_mul.defvjp(lambda ans, vs, gvs, vs_, x, a: lambda g:
                     vs.covector(gvs.scalar_mul(gvs.covector(g), a)), argnum=1)
vs_scalar_mul.defvjp(lambda ans, vs, gvs, vs_, x, a: lambda g:
                     gvs.inner_prod(g, gvs.covector(x)), argnum=2)

@primitive
def vs_inner_prod(vs, x, y):
    return vs._inner_prod(x, y)
vs_inner_prod.defvjp(lambda ans, vs, gvs, vs_, x, y: lambda g:
                     vs.covector(vs.scalar_mul(y, gvs.covector(g))), argnum=1)
vs_inner_prod.defvjp(lambda ans, vs, gvs, vs_, x, y: lambda g:
                     vs.covector(vs.scalar_mul(x, gvs.covector(g))), argnum=2)

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

class VSpace(object):
    __slots__ = []
    iscomplex = False
    def __init__(self, value):
        pass

    def zeros(self):          assert False
    def ones(self):           assert False
    def standard_basis(self): assert False
    def randn(self):          assert False

    add        = vs_add
    mut_add    = vs_mut_add
    scalar_mul = vs_scalar_mul
    inner_prod = vs_inner_prod
    covector   = vs_covector

    def _add(self, x, y):
        return x + y

    def _mut_add(self, x, y):
        x += y
        return x

    def _covector(self, x):
        return x

    def _scalar_mul(self, x, a):
        return x * a

    def _inner_prod(self, x, y):
        assert False

    def __eq__(self, other):
        return type(self) == type(other) and self.__dict__ == other.__dict__

    def __repr__(self):
        return "{}_{}".format(type(self).__name__, self.__dict__)

    def examples(self):
        # Used for testing only
        N = self.size
        unit_vect = np.zeros(N)
        unit_vect[npr.randint(N)] = 1.0
        unit_vect = self.unflatten(unit_vect)
        rand_vect = npr.randn(N)
        return [self.zeros(), self.unflatten(npr.randn(N))]

def vspace_flatten(value, covector=False):
    return vspace(value).flatten(value, covector)

box_type_mappings = {}
vspace_mappings = {}
box_types = set()
def register_box(box_type, value_type):
    box_types.add(box_type)
    box_type_mappings[value_type] = box_type
    box_type_mappings[box_type] = box_type

def register_vspace(vspace_maker, value_type):
    vspace_mappings[value_type] = vspace_maker

def new_box(value, node):
    try:
        return box_type_mappings[type(value)](value, node)
    except KeyError:
        raise TypeError("Can't differentiate w.r.t. type {}".format(type(value)))

def vspace(value):
    try:
        return vspace_mappings[type(value)](value)
    except KeyError:
        if isbox(value):
            return value.vspace
        else:
            raise TypeError("Can't find vspace for type {}".format(type(value)))

def subvals(x, ivs):
    x_ = list(x)
    for i, v in ivs:
        x_[i] = v
    return tuple(x_)

class SparseObject(object):
    __slots__ = ['vs', 'mut_add']
    def __init__(self, vs, mut_add):
        self.vs = vs
        self.mut_add = mut_add

register_vspace(lambda x : x.vs, SparseObject)
register_box(Box, SparseObject)

def assert_vspace_match(x, expected_vspace):
    assert expected_vspace == vspace(x), \
        "\nGrad returned unexpected vector space" \
        "\nVector space is {}" \
        "\nExpected        {}".format(vspace(x), expected_vspace)

isbox = lambda x: type(x) in box_types
getval = lambda x: getval(x.value) if isbox(x) else x
