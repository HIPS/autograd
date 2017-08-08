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
        if not isbox(end_box) or start_box not in end_box.node.progenitors:
            warnings.warn("Output seems independent of input.")
            def vjp(g): return start_box.vspace.zeros()
        else:
            def vjp(g): return backward_pass(g, end_box.node, start_box)
        return vjp, end_box
    return vjp_maker

def forward_pass(fun, args, kwargs, argnum=0):
    args = list(args)
    start_box = new_progenitor(args[argnum])
    args[argnum] = start_box
    active_progenitors.add(start_box)
    end_box = fun(*args, **kwargs)
    active_progenitors.remove(start_box)
    return start_box, end_box

def backward_pass(g, end_node, start_box):
    outgrads = {end_node : (g, False)}
    assert_vspace_match(outgrads[end_node][0], end_node.vspace, None)
    for node in toposort(end_node, start_box):
        if node not in outgrads: continue
        cur_outgrad = outgrads.pop(node)
        function, args, kwargs, parents, ans = node.recipe
        for argnum, parent in parents:
            outgrad = function.vjp(argnum, cur_outgrad[0], ans,
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
            if isbox(arg):
                argvals[argnum] = arg.value
                if argnum in self.zero_vjps: continue
                parents.append((argnum, arg.node))
                progenitors.update(arg.node.progenitors & active_progenitors)

        result_value = self.fun(*argvals, **kwargs)
        if progenitors:
            recipe = [self, args, kwargs, parents]
            result_box = new_box(result_value, recipe, progenitors)
            recipe.append(result_box)
            return result_box
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
    if isbox(x):
        box = new_box(x.value, (identity, (x,), {}, [(0, x.node)], x),
                        x.node.progenitors)
    else:
        box = new_box(x,       (identity, (x,), {}, [], x), set())
    box.node.progenitors = box.node.progenitors | {box}
    return box

@primitive
def identity(x) : return x
identity.defvjp(lambda g, ans, vs, gvs, x : g)

class Node(object):
    __slots__ = ['recipe', 'progenitors', 'vspace']

    def __init__(self, recipe, progenitors, vspace):
        self.recipe = recipe
        self.progenitors = progenitors
        self.vspace = vspace

class Box(object):
    __slots__ = ['vspace', 'value', 'node']

    def __init__(self, value, node):
        self.value = value
        self.node = node
        self.vspace = node.vspace

    def __bool__(self):
        return bool(self.value)

    __nonzero__ = __bool__

    def __str__(self):
        return "Autograd {0} with value {1} and {2} progenitors(s)".format(
            type(self).__name__, str(self.value), len(self.progenitors))

def toposort(end_node, start_box):
    def relevant_parents(node):
        return [parent for _, parent in node.recipe[3] if start_box in parent.progenitors]

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

box_type_mappings = {}
vspace_mappings = {}
box_types = set()
def register_box(box_type, value_type):
    box_types.add(box_type)
    box_type_mappings[value_type] = box_type
    box_type_mappings[box_type] = box_type

def register_vspace(vspace_maker, value_type):
    vspace_mappings[value_type] = vspace_maker

def new_box(value, recipe, progenitors):
    try:
        node = Node(recipe, progenitors, vspace(value))
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

class SparseObject(object):
    __slots__ = ['vs', 'mut_add']
    def __init__(self, vs, mut_add):
        self.vs = vs
        self.mut_add = mut_add

register_vspace(lambda x : x.vs, SparseObject)
register_box(Box, SparseObject)

def assert_vspace_match(x, expected_vspace, fun):
    assert expected_vspace == vspace(x), \
        "\nGrad of {} returned unexpected vector space" \
        "\nVector space is {}" \
        "\nExpected        {}".format(fun, vspace(x), expected_vspace)

isbox = lambda x: type(x) in box_types
getval = lambda x: x.value if isbox(x) else x

def unbox_if_possible(box):
    if isbox(box) and not active_progenitors.intersection(box.node.progenitors):
        return box.value
    return box
