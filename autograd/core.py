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
        start_node, end_node = forward_pass(fun, args, kwargs, argnum)
        if not isnode(end_node) or start_node not in end_node.progenitors:
            warnings.warn("Output seems independent of input.")
            def vjp(g): return start_node.vspace.zeros()
        else:
            def vjp(g): return backward_pass(g, end_node, start_node)
        return vjp, end_node
    return vjp_maker

def forward_pass(fun, args, kwargs, argnum=0):
    args = list(args)
    start_node = new_progenitor(args[argnum])
    args[argnum] = start_node
    active_progenitors.add(start_node)
    end_node = fun(*args, **kwargs)
    active_progenitors.remove(start_node)
    return start_node, end_node

def backward_pass(g, end_node, start_node):
    outgrads = {end_node : (g, False)}
    assert_vspace_match(outgrads[end_node][0], end_node.vspace, None)
    for node in toposort(end_node, start_node):
        cur_outgrad = outgrads.pop(node)
        function, args, kwargs, parents = node.recipe
        for argnum, parent in parents:
            outgrad = function.vjp(argnum, cur_outgrad[0], node,
                                   parent.vspace, node.vspace, args, kwargs)
            assert_vspace_match(outgrad, parent.vspace, function)
            outgrads[parent] = add_outgrads(parent.vspace, outgrads.get(parent), outgrad)
    return cur_outgrad[0]

def add_outgrads(vspace, prev_g_flagged, g):
    if prev_g_flagged is None:
        if type(getval(g)) == SparseObject:
            return primitive_mut_add(vspace, None, g), True
        else:
            return g, False
    else:
        prev_g, mutable = prev_g_flagged
        if mutable:
            return primitive_mut_add(vspace, prev_g, g), True
        else:
            prev_g_mutable = primitive_mut_add(vspace, None, prev_g)
            return primitive_mut_add(vspace, prev_g_mutable, g), True

active_progenitors = set()

class primitive(object):
    """
    Wraps a function so that its gradient can be specified and its invocation
    can be recorded. For examples, see the docs."""
    def __init__(self, fun):
        self.fun = fun
        self.vjps = {}
        self.zero_vjps = set()
        self.__name__ = fun.__name__
        self.__doc__ = fun.__doc__

    def __call__(self, *args, **kwargs):
        argvals = list(args)
        progenitors = set()
        parents = []
        for argnum, arg in enumerate(args):
            if isnode(arg):
                argvals[argnum] = arg.value
                if argnum in self.zero_vjps: continue
                parents.append((argnum, arg))
                progenitors.update(arg.progenitors & active_progenitors)

        result_value = self.fun(*argvals, **kwargs)
        if progenitors:
            return new_node(result_value, (self, args, kwargs, parents), progenitors)
        else:
            return result_value

    def vjp(self, argnum, outgrad, ans, vs, gvs, args, kwargs):
        try:
            return self.vjps[argnum](outgrad, ans, vs, gvs, *args, **kwargs)
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
            self.zero_vjps.add(argnum)

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
        def vjp(g, ans, vs, gvs, *args, **kwargs):
            return gradfun(ans, *args, **kwargs)(g)
        self.defvjp(vjp, argnum)

class nograd_primitive(primitive):
    def __call__(self, *args, **kwargs):
        argvals = map(getval, args)
        return self.fun(*argvals, **kwargs)

@primitive
def primitive_mut_add(vspace, x_prev, x_new):
    if x_prev is None:
        x_prev = vspace.zeros()
    if type(x_new) == SparseObject:
        return x_new.mut_add(x_prev)
    else:
        return vspace.mut_add(x_prev, x_new)
primitive_mut_add.vjp = lambda argnum, g, *args : g

def new_progenitor(x):
    if isnode(x):
        node = new_node(x.value, (identity, (x,), {}, [(0, x)]), x.progenitors)
    else:
        node = new_node(x,       (identity, (x,), {}, []      ), set())
    node.progenitors = node.progenitors | {node}
    return node

@primitive
def identity(x) : return x
identity.defvjp(lambda g, ans, vs, gvs, x : g)

class Node(object):
    __slots__ = ['value', 'recipe', 'progenitors', 'vspace']

    def __init__(self, value, recipe, progenitors):
        self.value = value
        self.recipe = recipe
        self.progenitors = progenitors
        self.vspace = vspace(value)

    def __bool__(self):
        return bool(self.value)

    __nonzero__ = __bool__

    def __str__(self):
        return "Autograd {0} with value {1} and {2} progenitors(s)".format(
            type(self).__name__, str(self.value), len(self.progenitors))

def toposort(end_node, start_node):
    def relevant_parents(node):
        return [parent for _, parent in node.recipe[3] if start_node in parent.progenitors]

    child_counts = {}
    stack = [end_node]
    while stack:
        node = stack.pop()
        if node in child_counts:
            child_counts[node] += 1
        else:
            child_counts[node] = 1
            stack.extend(relevant_parents(node))

    childless_nodes = [end_node]
    while childless_nodes:
        node = childless_nodes.pop()
        yield node
        for parent in relevant_parents(node):
            if child_counts[parent] == 1:
                childless_nodes.append(parent)
            else:
                child_counts[parent] -= 1

class VSpace(object):
    __slots__ = []
    iscomplex = False
    def __init__(self, value):
        pass

    def zeros(self):
        assert False

    def ones(self):
        assert False

    def standard_basis(self):
        assert False

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

node_type_mappings = {}
vspace_mappings = {}
node_types = set()
def register_node(node_type, value_type):
    node_types.add(node_type)
    node_type_mappings[value_type] = node_type
    node_type_mappings[node_type] = node_type

def register_vspace(vspace_maker, value_type):
    vspace_mappings[value_type] = vspace_maker

def new_node(value, recipe, progenitors):
    try:
        return node_type_mappings[type(value)](value, recipe, progenitors)
    except KeyError:
        raise TypeError("Can't differentiate w.r.t. type {}".format(type(value)))

def vspace(value):
    try:
        return vspace_mappings[type(value)](value)
    except KeyError:
        if isnode(value):
            return value.vspace
        else:
            raise TypeError("Can't find vspace for type {}".format(type(value)))

class SparseObject(object):
    __slots__ = ['vs', 'mut_add']
    def __init__(self, vs, mut_add):
        self.vs = vs
        self.mut_add = mut_add

register_vspace(lambda x : x.vs, SparseObject)
register_node(Node, SparseObject)

def assert_vspace_match(x, expected_vspace, fun):
    assert expected_vspace == vspace(x), \
        "\nGrad of {} returned unexpected vector space" \
        "\nVector space is {}" \
        "\nExpected        {}".format(fun, vspace(x), expected_vspace)

isnode = lambda x: type(x) in node_types
getval = lambda x: x.value if isnode(x) else x

def unbox_if_possible(node):
    if isnode(node) and not active_progenitors.intersection(node.progenitors):
        return node.value
    return node
