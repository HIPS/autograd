from __future__ import absolute_import
import warnings
import operator as op
import types
import math
import numpy as np
from weakref import WeakKeyDictionary

def grad(fun, argnum=0):
    """
    Returns a function which computes the gradient of `fun` with respect to
    positional argument number `argnum`. The returned function takes the same
    arguments as `fun`, but returns the gradient instead. The gradient has
    the same type as the argument."""
    def gradfun(*args, **kwargs):
        tape = CalculationTape()
        arg_wrt = args[argnum]
        start_node = new_node(safe_type(getval(arg_wrt)), [tape])
        args = list(args)
        args[argnum] = merge_tapes(start_node, arg_wrt)
        end_node = fun(*args, **kwargs)
        if not isinstance(end_node, Node) or tape not in end_node.tapes:
            warnings.warn("Output seems independent of input. Returning zero gradient.")
            return zeros_like(start_node)
        elif not isinstance(end_node.value, float):
            raise TypeError("Can only take gradient of scalar-valued functions. "\
                "You asked for the gradient of a {0}.".format(type(end_node.value)))
        else:
            end_node.tapes[tape].outgrads = [1.0]
            op_list = list(tape)
            del tape
            while op_list:
                node = op_list.pop()
                if node.outgrads:
                    cur_outgrad = node.sum_outgrads()
                    # assert type(getval(cur_outgrad)) is type(node.value), \
                    #     "Wrong outgrad type {0}. Should be {1}"\
                    #     .format(type(getval(cur_outgrad)), type(node.value))
                    for gradfun, parent in node.parent_grad_ops:
                        parent.outgrads.append(gradfun(cur_outgrad))
            return cur_outgrad
    try:
        gradfun.__name__ = "grad_{fun}_wrt_argnum_{argnum}".format(fun=fun.__name__, argnum=argnum)
        gradfun.__doc__ = "Gradient of function {fun} with respect to argument number {argnum}. " \
                          "Has the same arguments as {fun} but the return value has type of" \
                          "argument {argnum}".format(fun=fun.__name__, argnum=argnum)
    except:
        pass
    return gradfun

class primitive(object):
    """
    Wraps a function so that its gradient can be specified and its invocation
    can be recorded. For examples, see the docs."""
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
        self.grads[argnum] = gradmaker

    def defgrad_is_zero(self, argnums=(0,)):
        for argnum in argnums:
            self.zero_grads.add(argnum)

    def __call__(self, *args, **kwargs):
        argvals = list(args)
        ops = []
        tapes = set()
        for i, arg in enumerate(args):
            if isinstance(arg, Node):
                argvals[i] = arg.value
                if i in self.zero_grads: continue
                for tape in arg.tapes.keys():
                    ops.append((tape, i, arg))
                    tapes.add(tape)

        result = self.fun(*argvals, **kwargs)
        if result is NotImplemented: return result
        if ops:
            result = new_node(result, tapes)
            for tape, argnum, parent in ops:
                gradfun = self.gradmaker(argnum, result, *args, **kwargs)
                rnode = result.tapes[tape]
                rnode.parent_grad_ops.append((gradfun, parent.tapes[tape]))
        return result

    def __get__(self, obj, objtype):
        return types.MethodType(self, obj, objtype)

@primitive
def merge_tapes(x, y): return x
merge_tapes.defgrad(lambda ans, x, y : lambda g : g)
merge_tapes.defgrad(lambda ans, x, y : lambda g : g, argnum=1)

def new_node(value, tapes=[]):
    try:
        return Node.type_mappings[type(value)](value, tapes)
    except KeyError:
        raise TypeError("Can't differentiate wrt {0}".format(type(value)))

def zeros_like(value):
    if isinstance(value, Node):
        return value.zeros_like(value)
    else:
        return Node.type_mappings[type(value)].zeros_like(value)

class ReverseNode(object):
    __slots__ = ['parent_grad_ops', 'outgrads', 'node_type', 'value']
    def __init__(self, node_type, value):
        self.parent_grad_ops = []
        self.outgrads = []
        self.node_type = node_type
        self.value = value

    def sum_outgrads(self):
        return self.node_type.sum_outgrads(self.outgrads, self.value)

class Node(object):
    __slots__ = ['value', 'tapes']
    type_mappings = {}
    def __init__(self, value, tapes):
        self.value = value
        self.tapes = WeakKeyDictionary()
        for tape in tapes:
            new_rnode = ReverseNode(type(self), value)
            tape.append(new_rnode)
            self.tapes[tape] = new_rnode

    @staticmethod
    def sum_outgrads(outgrads, selftype):
        return cast(sum(outgrads[1:], outgrads[0]), selftype)

@primitive
def cast(x, value):
    if isinstance(x, np.ndarray):
        x = x[()]
    if np.iscomplexobj(x) and not np.iscomplexobj(value):
        x = np.real(x)
    return type(value)(x)

cast.defgrad(lambda ans, x, typecaster: I)

getval = lambda x : x.value if isinstance(x, Node) else x

class CalculationTape(list):
    def __hash__(self):
        return id(self)

class FloatNode(Node):
    __slots__ = []
    @staticmethod
    def zeros_like(value):
        if np.iscomplexobj(getval(value)):
            return 0.0 + 0.0j
        else:
            return 0.0
Node.type_mappings[float] = FloatNode

def safe_type(value):
    if isinstance(value, int):
        warnings.warn("Casting int to float to handle differentiation.")
        return float(value)
    else:
        return value

differentiable_ops = ['__add__', '__sub__', '__mul__', '__pow__', '__div__', '__mod__',
                      '__neg__', '__radd__', '__rsub__', '__rmul__', '__rpow__',
                      '__rdiv__', '__rmod__']
nondifferentiable_ops = ['__eq__', '__ne__', '__gt__', '__ge__', '__lt__', '__le__',]
for float_op in differentiable_ops + nondifferentiable_ops:
    setattr(FloatNode, float_op, primitive(getattr(float, float_op)))

FloatNode.__dict__['__neg__'].defgrad(lambda ans, x : op.neg)

for comp_op in nondifferentiable_ops:
    FloatNode.__dict__[comp_op].defgrad_is_zero(argnums=(0, 1))

# These functions will get clobbered when autograd.numpy is imported.
# They're here to allow the use of autograd without numpy.
I = lambda g: g
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
FloatNode.__dict__['__mod__'].defgrad(lambda ans, x, y : I)
FloatNode.__dict__['__mod__'].defgrad(lambda ans, x, y : lambda g : -g * floor(x/y), argnum=1)

log = primitive(math.log)
log.defgrad(lambda ans, x : lambda g : g / x)
floor = primitive(math.floor)
floor.defgrad_is_zero()

def swap_args(grads):
    grad_0, grad_1 = grads[1], grads[0]
    return {0 : lambda ans, y, x : grad_0(ans, x, y),
            1 : lambda ans, y, x : grad_1(ans, x, y)}

FloatNode.__dict__['__radd__'].grads = swap_args(FloatNode.__dict__['__add__'].grads)
FloatNode.__dict__['__rmul__'].grads = swap_args(FloatNode.__dict__['__mul__'].grads)
FloatNode.__dict__['__rsub__'].grads = swap_args(FloatNode.__dict__['__sub__'].grads)
FloatNode.__dict__['__rdiv__'].grads = swap_args(FloatNode.__dict__['__div__'].grads)
FloatNode.__dict__['__rpow__'].grads = swap_args(FloatNode.__dict__['__pow__'].grads)
FloatNode.__dict__['__rmod__'].grads = swap_args(FloatNode.__dict__['__mod__'].grads)
