import weakref
import warnings
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from operator import attrgetter
from itertools import count

def grad(fun, argnum=0):
    def gradfun(*args, **kwargs):
        tape = CalculationTape()
        start_node = Node(args[argnum], tape)
        args = args[:argnum] + (start_node,) + args[argnum+1:]
        end_node = fun(*args, **kwargs)
        if not tape.hasmember(end_node):
            warnings.warn("Output seems independent of input. Returning zero gradient.")
            return start_node.zeros()
        if not isinstance(getval(end_node), float):
            raise TypeError("Can only take gradient of scalar-valued functions")
        else:
            end_node.reverse_node.outgrads.append(1.0)
            for node in tape[::-1]:
                node.send_upstream()
            return start_node.reverse_node.sum_outgrads()

    return gradfun

def Differentiable(fun, forward_pass):
    def differentiable_fun(*args, **kwargs):
        tape = top_tape(args)
        if tape is None:
            return fun(*args, **kwargs)
        else:
            arg_vals = [arg.value if tape.hasmember(arg) else arg for arg in args]
            result, gradfuns = forward_pass(*arg_vals, **kwargs)
            parent_ops = [(gradfuns[i], parent.reverse_node)
                          for i, parent in enumerate(args) if tape.hasmember(parent)]
            return Node(result, tape, parent_ops)
        differentiable_fun.__name__ = fun.__name__
    return differentiable_fun

def primitive(fun, gradmaker):
    def forward_pass(*args, **kwargs):
        ans = differentiable_fun(*args, **kwargs)
        return ans, gradmaker(ans, *args, **kwargs)
    differentiable_fun = Differentiable(fun, forward_pass)
    return differentiable_fun

class CalculationTape(list):
    tape_count = count(0)
    def __init__(self):
        super(CalculationTape, self).__init__([])
        self.priority = self.tape_count.next()

    def hasmember(self, x):
        return isinstance(x, Node) and x.tape is self

def top_tape(args):
    tapes = [node.tape for node in args if isinstance(node, Node)]
    return max(tapes, key=attrgetter('priority')) if tapes else None

class ReverseNode(object):
    def __init__(self, parent_ops):
        self.parent_ops = parent_ops
        self.outgrads = []

    def send_upstream(self):
        if self.outgrads:
            outgrad_sum = self.sum_outgrads()
            for gradfun, parent in self.parent_ops:
                parent.add_outgrad(gradfun(outgrad_sum))

    def sum_outgrads(self):
        outgrad_sum = self.outgrads[0]
        for new in self.outgrads[1:]:
            outgrad_sum = outgrad_sum + new
        return outgrad_sum

    def add_outgrad(self, outgrad):
        self.outgrads.append(outgrad)

class Node(object):
    __slots__ = ['value', 'tape', 'parent_ops', 'outgrads']
    __metaclass__ = ABCMeta
    type_mappings = {}
    def __new__(cls, value, *args, **kwargs):
        try:
            subclass = Node.type_mappings[type(value)]
            return super(Node, cls).__new__(subclass, value, *args, **kwargs)
        except KeyError:
            raise TypeError("Can't differentiate wrt {0}".format(type(value)))

    def __init__(self, value, tape, parent_ops=[]):
        self.value = value
        self.tape = tape
        self.reverse_node = ReverseNode(parent_ops)
        tape.append(self.reverse_node)

    def __getitem__(self, idx):
        return take(self, idx)

    @abstractmethod
    def zeros(self):
        pass

    @staticmethod
    def add_subclass(subclass, value_types):
        Node.type_mappings[subclass] = subclass
        for value_type in value_types:
            Node.type_mappings[value_type] = subclass

def getval(x):
    return getval(x.value) if isinstance(x, Node) else x

def zeros_like(x):
    return Node(x, CalculationTape()).zeros()

def take(A, idx): return A[idx]
take = primitive(take, lambda ans, A, idx : [lambda g : untake(g, idx, lambda : zeros_like(A))])

def untake(x, idx, zeros):
    result = zeros()
    result[idx] = x
    return result
untake = primitive(untake, lambda ans, x, idx, zeros : [lambda g : take(g, idx)])
