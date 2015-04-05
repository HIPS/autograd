from __future__ import absolute_import
import warnings
import operator as op
import types
import numpy as np

def grad(fun, argnum=0, return_function_value=False):
    def gradfun(*args, **kwargs):
        tape = CalculationTape()
        start_node = args[argnum]
        if not isinstance(start_node, Node):
            start_node = safe_type(start_node)
            start_node = new_node(start_node)
        tape.add_node(start_node)
        args = args[:argnum] + (start_node,) + args[argnum+1:]
        end_node = fun(*args, **kwargs)
        tape.complete = True
        if not isinstance(end_node, Node) or tape not in end_node.tapes:
            warnings.warn("Output seems independent of input. Returning zero gradient.")
            gradval = zeros_like(start_node)
        elif not isinstance(end_node.value, float):
            raise TypeError("Can only take gradient of scalar-valued functions")
        else:
            end_node.tapes[tape].outgrads = [1.0]
            op_list = tape.op_list
            while op_list:
                node = op_list.pop()
                if node.outgrads:
                    cur_outgrad = node.sum_outgrads()
                    for gradfun, parent in node.parent_grad_ops:
                        parent.outgrads.append(gradfun(cur_outgrad))
            gradval = cur_outgrad
        if return_function_value:
            return getval(end_node), gradval
        else:
            return gradval
    return gradfun

class primitive(object):
    def __init__(self, fun):
        self.fun = fun
        self.grads = {}
        self.zero_grads = set()
        self.__name__ = fun.__name__
        self.__doc__ = fun.__doc__

    def gradmaker(self, argnum, *args, **kwargs):
        try:
            return self.grads[argnum](*args, **kwargs)
        except KeyError:
            raise NotImplementedError("Gradient of {0} w.r.t. arg number {1} not yet implemented".format(self.fun, argnum))

    def defgrad(self, gradmaker, argnum=0):
        gradmaker.__name__ = "grad_{0}_{1}".format(argnum, self.fun.__name__)
        self.grads[argnum] = gradmaker

    def defgrad_is_zero(self, argnums=(0,)):
        for argnum in argnums:
            self.zero_grads.add(argnum)

    def __call__(self, *args, **kwargs):
        argvals = list(args)
        ops = []
        for i, arg in enumerate(args):
            if isinstance(arg, Node):
                argvals[i] = arg.value
                if i in self.zero_grads: continue
                for tape in arg.tapes.keys():
                    if not tape.complete:
                        ops.append((tape, i, arg))
                    else:
                        del arg.tapes[tape]

        result = self.fun(*argvals, **kwargs)
        if result is NotImplemented: return result
        if ops:
            result = new_node(result)
            for tape, argnum, parent in ops:
                if tape not in result.tapes:
                    tape.add_node(result)
            for tape, argnum, parent in ops:
                gradfun = self.gradmaker(argnum, result, *args, **kwargs)
                rnode = result.tapes[tape]
                rnode.parent_grad_ops.append((gradfun, parent.tapes[tape]))
        return result

    def __get__(self, obj, objtype):
        return types.MethodType(self, obj, objtype)

def new_node(value):
    try:
        return Node.type_mappings[type(value)](value)
    except KeyError:
        raise TypeError("Can't differentiate wrt {0}".format(type(value)))

def zeros_like(value):
    if isinstance(value, Node):
        return value.zeros_like(value)
    else:
        return Node.type_mappings[type(value)].zeros_like(value)

class ReverseNode(object):
    __slots__ = ['parent_grad_ops', 'outgrads', 'node_type']
    def __init__(self, node_type):
        self.parent_grad_ops = []
        self.outgrads = []
        self.node_type = node_type

    def sum_outgrads(self):
        return self.node_type.sum_outgrads(self.outgrads)

class Node(object):
    __slots__ = ['value', 'tapes']
    type_mappings = {}
    def __init__(self, value):
        self.value = value
        self.tapes = {}

    @staticmethod
    def sum_outgrads(outgrads):
        return sum(outgrads[1:], outgrads[0])

getval = lambda x : x.value if isinstance(x, Node) else x

class CalculationTape(object):
    def __init__(self):
        self.op_list = []
        self.complete = False

    def add_node(self, node):
        new_rnode = ReverseNode(type(node))
        self.op_list.append(new_rnode)
        node.tapes[self] = new_rnode
        return new_rnode

class FloatNode(Node):
    __slots__ = []
    @staticmethod
    def zeros_like(value):
        return 0.0

float_types = [float, np.float64, np.float32, np.float16]
for ft in float_types:
    Node.type_mappings[ft] = FloatNode

def safe_type(value):
    if isinstance(value, int):
        warnings.warn("Casting int to float to handle differentiation.")
        return float(value)
    else:
        return value

differentiable_ops = ['__add__', '__sub__', '__mul__', '__pow__', '__div__',
                      '__neg__', '__radd__', '__rsub__', '__rmul__', '__rpow__', '__rdiv__']
nondifferentiable_ops = ['__eq__', '__ne__', '__gt__', '__ge__', '__lt__', '__le__',]
for float_op in differentiable_ops + nondifferentiable_ops:
    setattr(FloatNode, float_op, primitive(getattr(float, float_op)))

FloatNode.__dict__['__neg__'].defgrad(lambda ans, x : op.neg)

for comp_op in nondifferentiable_ops:
    FloatNode.__dict__[comp_op].defgrad_is_zero(argnums=(0, 1))

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

log = primitive(np.log)
log.defgrad(lambda ans, x : lambda g : g / x)
