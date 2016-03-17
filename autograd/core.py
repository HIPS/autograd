from __future__ import absolute_import, print_function
import sys
import warnings
import copy
import operator as op
import types
import math
import re
import numpy as np
from functools import partial
from future.utils import iteritems, raise_from, raise_
from collections import defaultdict

def grad(fun, argnum=0):
    """
    Returns a function which computes the gradient of `fun` with respect to
    positional argument number `argnum`. The returned function takes the same
    arguments as `fun`, but returns the gradient instead. The function `fun`
    should be scalar-valued. The gradient has the same type as the argument."""
    @attach_name_and_doc(fun, argnum, 'Gradient')
    def gradfun(*args,**kwargs):
        return backward_pass(*forward_pass(fun,args,kwargs,argnum))
    return gradfun

def forward_pass(fun, args, kwargs, argnum=0):
    tape = CalculationTape()
    arg_wrt = args[argnum]
    start_node = new_node(safe_type(getval(arg_wrt)), None, None, None, set([tape]))
    args = list(args)
    args[argnum] = merge_tapes(start_node, arg_wrt)
    try: end_node = fun(*args, **kwargs)
    except Exception as e: add_extra_error_message(e)
    return start_node, end_node, tape

def backward_pass(start_node, end_node, tape):
    if not isinstance(end_node, Node) or tape not in end_node.tapes:
        warnings.warn("Output seems independent of input. Returning zero gradient.")
        return zeros_like(start_node)
    if type(end_node) is not FloatNode:
        try:
            end_node = FloatNode.cast(end_node, 1.0)
        except TypeError:
            raise TypeError(
                "Output type {} can't be cast to float. "
                "Function grad requires a scalar-valued function. "
                "Try jacobian or elementwise_grad.".format(type(end_node.value)))

    outgrads = defaultdict(list)
    outgrads[end_node] = [1.0]
    tape.complete = True
    for node in tape[::-1]:
        if node in outgrads:
            cur_outgrad = node.node_type.sum_outgrads(outgrads[node])
            assert type(new_node(getval(cur_outgrad))) == node.node_type, \
                "Types are {0} and {1}".format(type(new_node(getval(cur_outgrad))),
                                               node.node_type)
            for argnum, parent in enumerate(node.args):
                if isinstance(parent, Node) and argnum not in node.function.zero_grads:
                    gradfun = node.function.gradmaker(
                        argnum, node, node.args, node.kwargs)
                    og = cast_to_node_type(gradfun(cur_outgrad),
                                           parent.node_type, parent.value)
                    outgrads[parent].append(og)

    return cur_outgrad

def attach_name_and_doc(fun, argnum, opname):
    namestr = "{op}_{fun}_wrt_argnum_{argnum}".format(
        op=opname.lower(), fun=fun.__name__, argnum=argnum)
    docstr = "{op} of function {fun} with respect to argument number {argnum}. " \
        "Has the same arguments as {fun} but the return value has type of" \
        "argument {argnum}".format(op=opname, fun=fun.__name__, argnum=argnum)

    def wrap(gradfun):
        try:
            gradfun.__name__ = namestr
            gradfun.__doc__ = docstr
        finally:
            return gradfun
    return wrap

def cast_to_node_type(x, node_type, example):
    if type(new_node(getval(x))) is not node_type:
        return node_type.cast(x, example)
    else:
        return x

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

    def gradmaker(self, argnum, ans, args, kwargs):
        try:
            return self.grads[argnum](ans, *args, **kwargs)
        except KeyError:
            def error(*args, **kwargs):
                if self.grads == {}:
                    errstr = "Gradient of {0} not yet implemented."
                else:
                    errstr = "Gradient of {0} w.r.t. arg number {1} not yet implemented."
                raise NotImplementedError(errstr.format(self.fun.__name__, argnum))
            return error

    def defgrad(self, gradmaker, argnum=0):
        self.grads[argnum] = gradmaker

    def defgrads(self, gradmaker, argnums):
        for argnum in argnums:
            self.defgrad(partial(gradmaker, argnum), argnum)

    def defgrad_is_zero(self, argnums=(0,)):
        for argnum in argnums:
            self.zero_grads.add(argnum)


    def __call__(self, *args, **kwargs):
        argvals = []
        tapes = set()
        for i, arg in enumerate(args):
            if isinstance(arg, Node):
                argvals.append(arg.value)
                if i not in self.zero_grads:
                    for tape in arg.tapes:
                        if not tape.complete:
                            tapes.add(tape)
            else:
                argvals.append(arg)

        result = self.fun(*argvals, **kwargs)
        if result is NotImplemented: return result
        if tapes:
            result = new_node(result, self, args, kwargs, tapes)
            for tape in tapes:
                tape.append(result)

        return result

    if sys.version_info >= (3,):
        def __get__(self, obj, objtype):
            return types.MethodType(self, obj)
    else:
        def __get__(self, obj, objtype):
            return types.MethodType(self, obj, objtype)

@primitive
def merge_tapes(x, y): return x
merge_tapes.defgrad(lambda ans, x, y : lambda g : g)
merge_tapes.defgrad(lambda ans, x, y : lambda g : g, argnum=1)

def new_node(value, *args):
    try:
        return Node.type_mappings[type(value)](value, *args)
    except KeyError:
        return NoDerivativeNode(value, *args)

def zeros_like(value):
    if isinstance(value, Node):
        return value.zeros_like(value)
    else:
        return new_node(value, []).zeros_like(value)

class Node(object):
    __slots__ = ['value', 'function', 'args', 'kwargs', 'tapes', 'node_type']
    type_mappings = {}
    def __init__(self, value, function=None, args=(), kwargs=None, tapes=None):
        self.value = value
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.tapes = tapes
        self.node_type = type(self)

    def __bool__(self):
        return bool(self.value)

    __nonzero__ = __bool__

    @staticmethod
    def sum_outgrads(outgrads):
        return sum(outgrads[1:], outgrads[0])

    def __str__(self):
        return "Autograd {0} with value {1} and {2} tape(s)".format(
            type(self).__name__, str(self.value), len(self.tapes))

@primitive
def cast(value, caster):
    return caster(value)
cast.defgrad(lambda *args: I)

getval = lambda x : x.value if isinstance(x, Node) else x

class CalculationTape(list):
    def __init__(self):
        self.complete = False

    def __hash__(self):
        return id(self)

class FloatNode(Node):
    __slots__ = []
    @staticmethod
    def zeros_like(value):
        return 0.0
    @staticmethod
    def cast(value, example):
        return cast(value, cast_to_float)

Node.type_mappings[float] = FloatNode

def cast_to_float(x):
    if np.iscomplexobj(x):
        x = np.real(x)
    return float(x)

class ComplexNode(FloatNode):
    @staticmethod
    def zeros_like(value):
        return 0.0 + 0.0j
    @staticmethod
    def cast(value, example):
        return cast(value, cast_to_complex)

def cast_to_complex(value):
    if isinstance(value, np.ndarray):
        return complex(value[()])
    else:
        return complex(value)
Node.type_mappings[complex] = ComplexNode

def safe_type(value):
    if isinstance(value, int):
        warnings.warn("Casting int to float to handle differentiation.")
        return float(value)
    else:
        return value

if sys.version_info >= (3,):
    DIV = '__truediv__'
    RDIV = '__rtruediv__'
else:
    DIV = '__div__'
    RDIV = '__rdiv__'

differentiable_ops = ['__add__', '__sub__', '__mul__', '__pow__', '__mod__',
                      '__neg__', '__radd__', '__rsub__', '__rmul__', '__rpow__',
                      '__rmod__', DIV, RDIV]

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
FloatNode.__dict__[DIV].defgrad(lambda ans, x, y : lambda g : g / y)
FloatNode.__dict__[DIV].defgrad(lambda ans, x, y : lambda g : - g * x / y**2, argnum=1)
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
FloatNode.__dict__[RDIV].grads = swap_args(FloatNode.__dict__[DIV].grads)
FloatNode.__dict__['__rpow__'].grads = swap_args(FloatNode.__dict__['__pow__'].grads)
FloatNode.__dict__['__rmod__'].grads = swap_args(FloatNode.__dict__['__mod__'].grads)


# These two nodes are for handling errors. Instead of raising errors immediately
# on the forward pass, we build them into the graph and raise them on the
# reverse pass so that evaluating nondifferentiable functions that don't affect
# the output don't cause problems (c.f. Issue #43).

class NoDerivativeNode(FloatNode):
    # inherit from FloatNode so that numerical infix operators work
    @staticmethod
    def cast(value, example):
        return example  # pass through so we can raise an error on reverse pass


class AutogradHint(Exception):
    def __init__(self, message, subexception_type=None, subexception_val=None):
        self.message = message
        self.subexception_type = subexception_type
        self.subexception_val = subexception_val

    def __str__(self):
        if self.subexception_type:
            return '{message}\nSub-exception:\n{name}: {str}'.format(
                message=self.message,
                name=self.subexception_type.__name__,
                str=self.subexception_type(self.subexception_val))
        else:
            return self.message

common_errors = [
    ((TypeError, r'float() argument must be a string or a number'),
        "This error *might* be caused by assigning into arrays, which autograd doesn't support."),
    ((TypeError, r"got an unexpected keyword argument '(?:dtype)|(?:out)'" ),
        "This error *might* be caused by importing numpy instead of autograd.numpy. \n"
        "Check that you have 'import autograd.numpy as np' instead of 'import numpy as np'."),
]

def check_common_errors(error_type, error_message):
    keys, vals = zip(*common_errors)
    match = lambda key: error_type == key[0] and len(re.findall(key[1], error_message)) != 0
    matches = map(match, keys)
    num_matches = sum(matches)

    if num_matches == 1:
        return vals[matches.index(True)]

def add_extra_error_message(e):
    etype, value, traceback = sys.exc_info()
    extra_message = check_common_errors(type(e), str(e))

    if extra_message:
        if sys.version_info >= (3,):
            raise_from(AutogradHint(extra_message), e)
        else:
            raise_(AutogradHint, (extra_message, etype, value), traceback)
    raise_(etype, value, traceback)
