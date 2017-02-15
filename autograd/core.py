from __future__ import absolute_import
import sys
import types
import numpy as np
import numpy.random as npr
from functools import partial
from collections import defaultdict, OrderedDict
import warnings
from .errors import add_extra_error_message, defgrad_deprecated

def make_vjp(fun, argnum=0):
    def vjp(*args, **kwargs):
        start_node, end_node = forward_pass(fun, args, kwargs, argnum)
        if not isnode(end_node) or start_node not in end_node.progenitors:
            warnings.warn("Output seems independent of input.")
            return lambda g : start_node.vspace.zeros(), end_node
        return lambda g : backward_pass(g, end_node, start_node), end_node
    return vjp

def make_jvp(fun, argnum=0):
    def jvp(*args, **kwargs):
        args = list(args)
        start_node = new_progenitor(args[argnum])
        args[argnum] = start_node
        def forward_mode_pass(v):
            assert_vspace_match(v, start_node.vspace, None)
            start_node.forward_progenitors[start_node] = v
            active_forward_progenitors[start_node] = True
            end_node = fun(*args, **kwargs)
            active_forward_progenitors.pop(start_node)
            if not isnode(end_node) or start_node not in end_node.forward_progenitors:
                warnings.warn("Output seems independent of input.")
                return end_node, vspace(getval(end_node)).zeros()
            return end_node, end_node.forward_progenitors[start_node]
        return forward_mode_pass, start_node
    return jvp

def forward_pass(fun, args, kwargs, argnum=0):
    args = list(args)
    start_node = new_progenitor(args[argnum])
    args[argnum] = start_node
    active_progenitors.add(start_node)
    try: end_node = fun(*args, **kwargs)
    except Exception as e: add_extra_error_message(e)
    active_progenitors.remove(start_node)
    return start_node, end_node

def backward_pass(g, end_node, start_node):
    outgrads = defaultdict(list)
    outgrads[end_node] = [g]
    assert_vspace_match(outgrads[end_node][0], end_node.vspace, None)
    for node in toposort(end_node, start_node):
        if node not in outgrads: continue
        cur_outgrad = vsum(node.vspace, *outgrads[node])
        function, args, kwargs, parents = node.recipe
        for argnum, parent in parents:
            outgrad = function.vjp(argnum, cur_outgrad, node,
                                   parent.vspace, node.vspace, args, kwargs)
            outgrads[parent].append(outgrad)
            assert_vspace_match(outgrad, parent.vspace, function)
    return cur_outgrad

active_progenitors = set()
active_forward_progenitors = OrderedDict()

class primitive(object):
    """
    Wraps a function so that its gradient can be specified and its invocation
    can be recorded. For examples, see the docs."""
    def __init__(self, fun):
        self.fun = fun
        self.vjps = {}
        self.jvps = {}
        self.zero_vjps = set()
        self.__name__ = fun.__name__
        self.__doc__ = fun.__doc__

    def __call__(self, *args, **kwargs):
        argvals, parents, progenitors, forward_progenitors = self.find_progenitors(args)
        result_value = self.fun(*argvals, **kwargs)
        if progenitors or forward_progenitors:
            result = new_node(result_value, (self, args, kwargs, parents), progenitors, dict())
            if forward_progenitors:
                self.fwd_update(args, kwargs, result, forward_progenitors)
            return result
        else:
            return result_value

    def find_progenitors(self, args):
        argvals = list(args)
        parents = []
        progenitors = set()
        forward_progenitors = defaultdict(list)
        for argnum, arg in enumerate(args):
            if isnode(arg):
                argvals[argnum] = arg.value
                if argnum in self.zero_vjps: continue
                reverse = arg.progenitors & active_progenitors
                if reverse:
                    parents.append((argnum, arg))
                    progenitors.update(reverse)
                for progenitor in arg.forward_progenitors:
                    if active_forward_progenitors.get(progenitor, False):
                        forward_progenitors[progenitor].append((argnum, arg))
        return argvals, parents, progenitors, forward_progenitors

    def fwd_update(self, args, kwargs, result, forward_progenitors):
        for progenitor in forward_progenitors:
            active_forward_progenitors[progenitor] = False
        for progenitor in active_forward_progenitors:
            if progenitor not in forward_progenitors:
                continue
            ingrads = list()
            for argnum, arg in forward_progenitors[progenitor]:
                forward_grad = arg.forward_progenitors[progenitor]
                ingrad = self.jvp(argnum, forward_grad, result, arg.vspace,
                                  result.vspace, args, kwargs)
                assert_vspace_match(ingrad, result.vspace, self, fwd=True)
                ingrads.append(ingrad)
            result.forward_progenitors[progenitor] = vsum(result.vspace, *ingrads)
            active_forward_progenitors[progenitor] = True

    def vjp(self, argnum, outgrad, ans, vs, gvs, args, kwargs):
        try:
            return self.vjps[argnum](outgrad, ans, vs, gvs, *args, **kwargs)
        except KeyError:
            if self.vjps == {}:
                errstr = "Gradient of {0} not yet implemented."
            else:
                errstr = "Gradient of {0} w.r.t. arg number {1} not yet implemented."
            raise NotImplementedError(errstr.format(self.fun.__name__, argnum))

    def jvp(self, argnum, ingrad, ans, gvs, vs, args, kwargs):
        try:
            return self.jvps[argnum](ingrad, ans, gvs, vs, *args, **kwargs)
        except KeyError:
            if self.jvps == {}:
                errstr = "Forward gradient of {0} not yet implemented."
            else:
                errstr = "Forward gradient of {0} w.r.t. arg number {1} not yet implemented."
            raise NotImplementedError(errstr.format(self.fun.__name__, argnum))

    def defvjp(self, vjpmaker, argnum=0):
        vjpmaker.__name__ = "VJP_{}_of_{}".format(argnum, self.__name__)
        self.vjps[argnum] = vjpmaker

    def defjvp(self, jvpmaker, argnum=0):
        jvpmaker.__name__ = "JVP_{}_of_{}".format(argnum, self.__name__)
        self.jvps[argnum] = jvpmaker

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

def new_progenitor(x, fwd=False):
    if isnode(x):
        node = new_node(x.value, (identity, (x,), {}, [(0, x)]), x.progenitors, x.forward_progenitors)
    else:
        node = new_node(x,       (identity, (x,), {}, []      ), set(),         dict())
    if not fwd:
        node.progenitors = node.progenitors | {node}
    return node

def vsum(vspace, *args):
    if len(args) == 1 and type(getval(args[0])) != SparseObject:
        return args[0]
    else:
        return primitive_vsum(vspace, *args)

@primitive
def primitive_vsum(vspace, *args):
    ans = vspace.zeros()
    for arg in args:
        if type(arg) == SparseObject:
            ans = arg.mut_add(ans)
        else:
            ans = vspace.mut_add(ans, arg)
    return ans
primitive_vsum.vjp = lambda arg, g, *args : g
primitive_vsum.jvp = lambda arg, g, *args : g

@primitive
def identity(x) : return x
identity.defvjp(lambda g, ans, vs, gvs, x : g)

class Node(object):
    __slots__ = ['value', 'recipe', 'progenitors', 'forward_progenitors',
                 'vspace']

    def __init__(self, value, recipe, progenitors, forward_progenitors):
        self.value = value
        self.recipe = recipe
        self.progenitors = progenitors
        self.forward_progenitors = forward_progenitors
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

def new_node(value, recipe, progenitors, forward_progenitors):
    try:
        return node_type_mappings[type(value)](value, recipe, progenitors, forward_progenitors)
    except KeyError:
        raise TypeError("Can't differentiate w.r.t. type {}".format(type(value)))

def vspace(value):
    try:
        return vspace_mappings[type(value)](value)
    except KeyError:
        raise TypeError("Can't find vspace for type {}".format(type(value)))

class SparseObject(object):
    __slots__ = ['vs', 'mut_add']
    def __init__(self, vs, mut_add):
        self.vs = vs
        self.mut_add = mut_add

register_vspace(lambda x : x.vs, SparseObject)
register_node(Node, SparseObject)

def assert_vspace_match(x, expected_vspace, fun, fwd=False):
    grad_string = "Forward grad" if fwd else "Grad"
    assert expected_vspace == vspace(getval(x)), \
        "\n{} of {} returned unexpected vector space" \
        "\nVector space is {}" \
        "\nExpected        {}".format(grad_string, fun, vspace(getval(x)), expected_vspace)

isnode = lambda x: type(x) in node_types
getval = lambda x: x.value if isnode(x) else x
