from __future__ import absolute_import
import warnings
import operator as op
import types
from numpy import log, float64, ndarray

def grad(fun, argnum=0):
    def gradfun(*args, **kwargs):
        tape = CalculationTape()
        start_node = args[argnum]
        if not isinstance(start_node, Node):
            start_node = new_node(start_node)
        tape.add_node(start_node)
        args = args[:argnum] + (start_node,) + args[argnum+1:]
        end_node = fun(*args, **kwargs)
        tape.active = False
        if not isinstance(end_node, Node) or tape not in end_node.tapes:
            warnings.warn("Output seems independent of input. Returning zero gradient.")
            return 0 * start_node.value
        elif not isinstance(end_node.value, float):
            raise TypeError("Can only take gradient of scalar-valued functions")
        else:
            end_node.tapes[tape].outgrad = 1.0
            op_list = tape.op_list
            while op_list:
                node = op_list.pop()
                if node.outgrad is not 0:
                    for gradfun, parent in node.parent_ops:
                        parent.outgrad = parent.outgrad + gradfun(node.outgrad)
            return node.outgrad
    return gradfun

class primitive(object):
    def __init__(self, fun):
        self.fun = fun
        self.grads = {}

    def gradmaker(self, argnum, *args, **kwargs):
        try:
            return self.grads[argnum](*args, **kwargs)
        except KeyError:
            raise NotImplementedError("Gradient of {0} not yet implemented".format(self.fun))

    def defgrad(self, gradmaker, argnum=0):
        gradmaker.__name__ = "grad_{0}_{1}".format(argnum, self.fun.__name__)
        self.grads[argnum] = gradmaker

    def __call__(self, *args, **kwargs):
        argvals = list(args)
        ops = []
        for i, arg in enumerate(args):
            if isinstance(arg, Node):
                argvals[i] = arg.value
                for tape in arg.tapes.keys():
                    if tape.active:
                        ops.append((tape, i, arg))
                    else:
                        del arg.tapes[tape]

        result = self.fun(*argvals, **kwargs)
        assert not type(result) == ndarray, self.fun # Check for gaps in numpy wrapping
        if result is NotImplemented: return result
        if ops:
            result = new_node(result)
            for tape, argnum, parent in ops:
                if tape not in result.tapes:
                    rnode = tape.add_node(result)
                else:
                    rnode = result.tapes[tape]
                gradfun = self.gradmaker(argnum, result, *args, **kwargs)
                rnode.parent_ops.append((gradfun, parent.tapes[tape]))
        return result

    def __get__(self, obj, objtype):
        return types.MethodType(self, obj, objtype)

def new_node(value):
    try:
        return Node.type_mappings[type(value)](value)
    except KeyError:
        raise TypeError("Can't differentiate wrt {0}".format(type(value)))

class Node(object):
    __slots__ = ['value', 'tapes']
    type_mappings = {}
    def __init__(self, value):
        self.value = value
        self.tapes = {}

class ReverseNode(object):
    __slots__ = ['parent_ops', 'outgrad']
    def __init__(self):
        self.parent_ops = []
        self.outgrad = 0

class CalculationTape(object):
    def __init__(self):
        self.op_list = []
        self.active = True

    def add_node(self, node):
        new_rnode = ReverseNode()
        self.op_list.append(new_rnode)
        node.tapes[self] = new_rnode
        return new_rnode

P = primitive
class FloatNode(Node):
    __slots__ = []
    __add__  = P(float.__add__)
    __sub__  = P(float.__sub__)
    __mul__  = P(float.__mul__)
    __pow__  = P(float.__pow__)
    __div__  = P(float.__div__)
    __neg__  = P(float.__neg__)
    __radd__ = P(float.__radd__)
    __rsub__ = P(float.__rsub__)
    __rmul__ = P(float.__rmul__)
    __rpow__ = P(float.__rpow__)
    __rdiv__ = P(float.__rdiv__)
Node.type_mappings[float] = FloatNode
Node.type_mappings[float64] = FloatNode

I = lambda x : x
FloatNode.__dict__['__neg__'].defgrad(lambda ans, x : op.neg)

FloatNode.__dict__['__add__'].defgrad(lambda ans, x, y : I)
FloatNode.__dict__['__add__'].defgrad(lambda ans, x, y : I, argnum=1)
FloatNode.__dict__['__mul__'].defgrad(lambda ans, x, y : lambda g : y * g)
FloatNode.__dict__['__mul__'].defgrad(lambda ans, x, y : lambda g : x * g, argnum=1)
FloatNode.__dict__['__sub__'].defgrad(lambda ans, x, y : I)
FloatNode.__dict__['__sub__'].defgrad(lambda ans, x, y : op.neg, argnum=1)
FloatNode.__dict__['__div__'].defgrad(lambda ans, x, y : lambda g : g / y)
FloatNode.__dict__['__div__'].defgrad(lambda ans, x, y : lambda g : - g * x / y**2, argnum=1)
FloatNode.__dict__['__pow__'].defgrad(lambda ans, x, y : lambda g : g * y * x ** (y - 1))
FloatNode.__dict__['__pow__'].defgrad(lambda ans, x, y : lambda g : g * log(x) * x ** y, argnum=1)

def swap_args(grads):
    grad_0, grad_1 = grads[1], grads[0]
    return {0 : lambda ans, y, x : grad_0(ans, x, y),
            1 : lambda ans, y, x : grad_1(ans, x, y)}

FloatNode.__dict__['__radd__'].grads = swap_args(FloatNode.__dict__['__add__'].grads)
FloatNode.__dict__['__rmul__'].grads = swap_args(FloatNode.__dict__['__mul__'].grads)
FloatNode.__dict__['__rsub__'].grads = swap_args(FloatNode.__dict__['__sub__'].grads)
FloatNode.__dict__['__rdiv__'].grads = swap_args(FloatNode.__dict__['__div__'].grads)
FloatNode.__dict__['__rpow__'].grads = swap_args(FloatNode.__dict__['__pow__'].grads)

log = P(log)
log.defgrad(lambda ans, x : lambda g : g / x)
