from __future__ import absolute_import
import sys
import types
import numpy as np
import numpy.random as npr
from functools import partial
import warnings
from .errors import defgrad_deprecated

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
    start_box = new_box(args[argnum], new_trace())
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
        boxed_args, trace = find_top_boxed_args(args)
        if boxed_args:
            argvals = subvals(args, [(argnum, box.value) for argnum, box in boxed_args])
            result = self(*argvals, **kwargs)
            parents = [box.node for _, box in boxed_args]
            vjps = [self.vjp(argnum, result, box.node.vspace,
                             vspace(result), argvals, kwargs)
                    for argnum, box in boxed_args]
            return new_box(result, trace, parents, vjps)
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
                return vs_sparse_add(vs, prev_g, g), True
            else:
                return vs_mut_add(vs, prev_g, g), True
        else:
            if sparse:
                prev_g_mutable = vs_mut_add(vs, vs.zeros(), prev_g)
                return vs_sparse_add(vs, prev_g_mutable, g), True
            else:
                return vs_add(vs, prev_g, g), True
    else:
        if sparse:
            return vs_sparse_add(vs, vs.zeros(), g), True
        else:
            return g, False

@primitive
def vs_add(vs, x_prev, x_new): return vs.add(x_prev, x_new)
vs_add.defvjps(identity_vjp, argnums=[1,2])

@primitive
def vs_sparse_add(vs, x_prev, x_new): return x_new.mut_add(x_prev)
vs_sparse_add.defvjps(identity_vjp, argnums=[1,2])

@primitive
def vs_mut_add(vs, x_prev, x_new): return vs.mut_add(x_prev, x_new)
vs_mut_add.defvjps(identity_vjp, argnums=[1,2])

def find_top_boxed_args(args):
    top_trace = -1
    top_boxes = []
    for argnum, arg in enumerate(args):
        if isbox(arg):
            trace = arg._trace
            if trace > top_trace:
                top_boxes = [(argnum, arg)]
                top_trace = trace
            elif trace == top_trace:
                top_boxes.append((argnum, arg))
    return top_boxes, top_trace

global_top_trace = 0
def new_trace():
    global global_top_trace
    global_top_trace += 1
    return global_top_trace

class Node(object):
    __slots__ = ['vspace', 'parents', 'vjps']
    def __init__(self, vspace, parents, vjps):
        self.vspace = vspace
        self.parents = parents
        self.vjps = vjps

class Box(object):
    __slots__ = ['vspace', 'value', '_trace', 'node']
    def __init__(self, value, trace, node):
        self.value = value
        self.node = node
        self._trace = trace
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

    def add(self, x, y):
        return x + y

    def mut_add(self, x, y):
        x += y
        return x

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

def new_box(value, trace, parents=(), vjps=()):
    try:
        node = Node(vspace(value), parents, vjps)
        return box_type_mappings[type(value)](value, trace, node)
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
