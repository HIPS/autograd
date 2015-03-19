from __future__ import absolute_import
import warnings
import operator as op
from operator import attrgetter
from itertools import count
from numpy import log

def grad(fun, argnum=0):
    def gradfun(*args, **kwargs):
        tape = CalculationTape()
        start_node = tape.add_node(args[argnum])
        args = args[:argnum] + (start_node,) + args[argnum+1:]
        end_node = fun(*args, **kwargs)
        tape.deactivate()
        if not tape.hasmember(end_node):
            warnings.warn("Output seems independent of input. Returning zero gradient.")
            return 0 * args[argnum]
        elif not isinstance(end_node, float):
            raise TypeError("Can only take gradient of scalar-valued functions")
        else:
            tape.lookup(end_node).outgrad = 1.0
            for node in tape[::-1]:
                if node.outgrad is not 0:
                    for gradfun, parent in node.parent_ops:
                        parent.outgrad = parent.outgrad + gradfun(node.outgrad)
            return tape[0].outgrad

    return gradfun

def grad_not_implemented(*args, **kwargs):
    raise NotImplementedError("Gradient not implemented yet")

def primitive(fun, gradmaker=grad_not_implemented):
    def wrapped_function(*args, **kwargs):
        result = fun(*args, **kwargs)
        all_parent_ops = []
        for tape in CalculationTape.active_tapes[::-1]:
            parent_ops = [(gradmaker(result, *args, **kwargs)[i], tape.lookup(parent))
                          for i, parent in enumerate(args) if tape.hasmember(parent)]
            if parent_ops:
                result = tape.add_node(result, parent_ops)
        return result
    return wrapped_function

class CalculationTape(list):
    tape_count = count(0)
    active_tapes = []
    def __init__(self):
        super(CalculationTape, self).__init__([])
        self.obj_lookup = {}
        self.priority = self.tape_count.next()
        self.active_tapes.append(self)

    def hasmember(self, x):
        return id(x) in self.obj_lookup

    def lookup(self, node):
        return self[self.obj_lookup[id(node)]]

    def add_node(self, node, parent_ops = []):
        if isinstance(node, float) and not isinstance(node, FloatNode):
            node = FloatNode(node)
        self.append(ReverseNode(node, parent_ops))
        self.obj_lookup[id(node)] = len(self) - 1
        return node

    def deactivate(self):
        for i, tape in enumerate(self.active_tapes):
            if tape is self:
                del(self.active_tapes[i])

def add_tape(node, tape, ops=None):
    if isinstance(node, float):
        node = FloatNode(node)
    return node.add_tape(tape, ops)

class ReverseNode(object):
    __slots__ = ['parent_ops', 'outgrad', 'node']
    def __init__(self, node, parent_ops):
        self.node = node
        self.parent_ops = parent_ops
        self.outgrad = 0

I = lambda x : x # Identity operator
grad_neg = lambda ans, x    : [neg]
grad_add = lambda ans, x, y : [I, I]
grad_mul = lambda ans, x, y : [lambda g : y * g, lambda g : x * g]
grad_sub = lambda ans, x, y : [I, neg]
grad_div = lambda ans, x, y : [lambda g : g / y, lambda g : - g * x / y**2]
grad_pow = lambda ans, x, y : [lambda g : g * y * x ** (y - 1),
                               lambda g : g * log(x) * x ** y]
grad_log = lambda ans, x    : [lambda g : g / x]
def reverse_args(fun):
    def new_fun(ans, x, y):
        return fun(ans, y, x)[::-1]
    return new_fun

P = primitive
class FloatNode(float):
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
