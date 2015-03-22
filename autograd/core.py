from __future__ import absolute_import
import warnings
import operator as op
from operator import attrgetter
from numpy import log, float64, ndarray
from collections import OrderedDict

def grad(fun, argnum=0):
    def gradfun(*args, **kwargs):
        tape = CalculationTape()
        start_node = new_node(args[argnum])
        tape.add_node(start_node)
        args = args[:argnum] + (start_node,) + args[argnum+1:]
        end_node = fun(*args, **kwargs)
        if end_node not in tape:
            warnings.warn("Output seems independent of input. Returning zero gradient.")
            return 0 * start_node.value
        elif not isinstance(end_node.value, float):
            raise TypeError("Can only take gradient of scalar-valued functions")
        else:
            tape[end_node].outgrad = 1.0
            tape.finalize()
            while tape:
                _, node = tape.popitem()
                if node.outgrad is not 0:
                    for gradfun, parent in node.parent_ops:
                        parent.outgrad += gradfun(node.outgrad)
            return node.outgrad
    return gradfun

getval = lambda x : x.value if isinstance(x, Node) else x

def primitive(fun, gradmaker):
    def wrapped_function(*args, **kwargs):
        result = fun(*map(getval, args), **kwargs)
        assert not type(result) == ndarray, fun # Check for gaps in numpy wrapping
        if result is NotImplemented:
            return result
        for i, arg in enumerate(args):
            if isinstance(arg, Node) and arg.tapes:
                result = new_node(result)
                gradfun = gradmaker(result, *args, **kwargs)[i]
                for tape in arg.tapes:
                    tape.add_operation(result, (gradfun, tape[arg]))
        return result
    wrapped_function.__name__ = fun.__name__
    return wrapped_function

def new_node(value):
    if isinstance(value, Node):
        return value
    try:
        return Node.type_mappings[type(value)](value)
    except KeyError:
        raise TypeError("Can't differentiate wrt {0}".format(type(value)))

class Node(object):
    __slots__ = ['value', 'tape']
    type_mappings = {}
    def __init__(self, value):
        self.value = value
        self.tapes = []

class ReverseNode(object):
    __slots__ = ['parent_ops', 'outgrad']
    def __init__(self):
        self.parent_ops = []
        self.outgrad = 0

    def remove_self_from_node(self, tape):
        del self.node.tapes[tape]

class CalculationTape(OrderedDict):
    def finalize(self):
        for node in self.keys():
            node.tapes.remove(self)

    def add_operation(self, node, reverse_op):
        if node not in self:
            self.add_node(node)
        self[node].parent_ops.append(reverse_op)

    def add_node(self, node):
        self[node] = ReverseNode()
        node.tapes.append(self)

I = lambda x : x
grad_neg = lambda ans, x    : [op.neg]
grad_add = lambda ans, x, y : [I, I]
grad_mul = lambda ans, x, y : [lambda g : y * g, lambda g : x * g]
grad_sub = lambda ans, x, y : [I, op.neg]
grad_div = lambda ans, x, y : [lambda g : g / y, lambda g : - g * x / y**2]
grad_pow = lambda ans, x, y : [lambda g : g * y * x ** (y - 1),
                               lambda g : g * log(x) * x ** y]
grad_log = lambda ans, x    : [lambda g : g / x]

log = primitive(log, grad_log)

def reverse_args(fun):
    def reversed_fun(ans, x, y):
        return fun(ans, y, x)[::-1]
    return reversed_fun

P = primitive
class FloatNode(Node):
    __add__  = P(float.__add__ , grad_add)
    __sub__  = P(float.__sub__,  grad_sub)
    __mul__  = P(float.__mul__,  grad_mul)
    __pow__  = P(float.__pow__,  grad_pow)
    __div__  = P(float.__div__,  grad_div)
    __neg__  = P(float.__neg__,  grad_neg)
    __radd__ = P(float.__radd__, reverse_args(grad_add))
    __rsub__ = P(float.__rsub__, reverse_args(grad_sub))
    __rmul__ = P(float.__rmul__, reverse_args(grad_mul))
    __rpow__ = P(float.__rpow__, reverse_args(grad_pow))
    __rdiv__ = P(float.__rdiv__, reverse_args(grad_div))
Node.type_mappings[float] = FloatNode
Node.type_mappings[float64] = FloatNode
