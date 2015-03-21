from __future__ import absolute_import
import warnings
import operator as op
from operator import attrgetter
from itertools import count
from numpy import log, float64, ndarray

def grad(fun, argnum=0):
    def gradfun(*args, **kwargs):
        tape = CalculationTape()
        start_node = new_node(args[argnum])
        start_node.reverse_nodes[tape] = ReverseNode(tape, start_node)
        args = args[:argnum] + (start_node,) + args[argnum+1:]
        end_node = fun(*args, **kwargs)
        if not (isinstance(end_node, Node) and tape in end_node.reverse_nodes):
            warnings.warn("Output seems independent of input. Returning zero gradient.")
            return 0 * start_node.value
        elif not isfloaty(end_node.value):
            raise TypeError("Can only take gradient of scalar-valued functions")
        else:
            end_node.reverse_nodes[tape].outgrad = 1.0
            tape.finalize()
            for node in tape[::-1]:
                if node.outgrad is not 0:
                    for gradfun, parent in node.parent_ops:
                        parent.outgrad = parent.outgrad + gradfun(node.outgrad)
            return tape[0].outgrad
    return gradfun

def isfloaty(x):
    return isinstance(x, float) or (isinstance(x, ndarray) and x.shape == ())

getval = lambda x : x.value if isinstance(x, Node) else x

def primitive(fun, gradmaker):
    def wrapped_function(*args, **kwargs):
        result = fun(*map(getval, args), **kwargs)
        assert not type(result) == ndarray, fun
        if result is NotImplemented:
            return result
        for i, arg in enumerate(args):
            if isinstance(arg, Node):
                for tape in arg.reverse_nodes:
                    result = new_node(result)
                    gradfun = gradmaker(result, *args, **kwargs)[i]
                    result.add_reverse_op(tape, (gradfun, arg.reverse_nodes[tape]))
        return result
    wrapped_function.__name__ = fun.__name__
    return wrapped_function

def new_node(value):
    if isinstance(value, Node):
        return value
    else:
        return Node(value)

class Node(object):
    type_mappings = {}
    def __new__(cls, value):
        try:
            subclass = Node.type_mappings[type(value)]
            return super(Node, cls).__new__(subclass, value)
        except KeyError:
            raise TypeError("Can't differentiate wrt {0}".format(type(value)))

    def __init__(self, value):
        self.value = value
        self.reverse_nodes = {}

    def add_reverse_op(self, tape, reverse_op):
        if tape not in self.reverse_nodes:
            self.reverse_nodes[tape] = ReverseNode(tape, self)
        self.reverse_nodes[tape].parent_ops.append(reverse_op)

class ReverseNode(object):
    __slots__ = ['parent_ops', 'outgrad', 'tape', 'node']
    def __init__(self, tape, node):
        self.tape = tape
        self.tape.append(self)
        self.parent_ops = []
        self.outgrad = 0
        self.node = node

    def remove_self_from_node(self):
        del self.node.reverse_nodes[self.tape]

class CalculationTape(list):
    def finalize(self):
        for rnode in self:
            rnode.remove_self_from_node()

    def __hash__(self):
        return id(self)

P = primitive
I = lambda x : x # Identity operator
grad_neg = lambda ans, x    : [op.neg]
grad_add = lambda ans, x, y : [I, I]
grad_mul = lambda ans, x, y : [lambda g : y * g, lambda g : x * g]
grad_sub = lambda ans, x, y : [I, op.neg]
grad_div = lambda ans, x, y : [lambda g : g / y, lambda g : - g * x / y**2]
grad_pow = lambda ans, x, y : [lambda g : g * y * x ** (y - 1),
                               lambda g : g * log(x) * x ** y]
grad_log = lambda ans, x    : [lambda g : g / x]
log = P(log, grad_log)

def reverse_args(fun):
    def new_fun(ans, x, y):
        return fun(ans, y, x)[::-1]
    return new_fun

class FloatNode(Node):
    __add__  = P(float.__add__ , grad_add)
    __radd__ = P(float.__radd__, reverse_args(grad_add))
    __sub__  = P(float.__sub__,  grad_sub)
    __rsub__ = P(float.__rsub__, reverse_args(grad_sub))
    __mul__  = P(float.__mul__,  grad_mul)
    __rmul__ = P(float.__rmul__, reverse_args(grad_mul))
    __pow__  = P(float.__pow__,  grad_pow)
    __rpow__ = P(float.__rpow__, reverse_args(grad_pow))
    __div__  = P(float.__div__,  grad_div)
    __rdiv__ = P(float.__rdiv__, reverse_args(grad_div))
    __neg__  = P(float.__neg__,  grad_neg)
Node.type_mappings[float] = FloatNode
Node.type_mappings[float64] = FloatNode
