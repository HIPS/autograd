import operator as op
import numpy as np
from functools import partial
from operator import attrgetter
from copy import copy

# ----- Autodiff logic -----

def grad(fun, argnum=0):
    def gradfun(*args):
        tape = CalculationTape(top_tape(args))
        start_node = new_node(args[argnum], tape)
        args = args[:argnum] + (start_node,) + args[argnum+1:]
        end_node = fun(*args)
        if not ismember(end_node, tape):
            return start_node.get_outgrad()
        elif not isinstance(end_node, scalarNode):
            raise TypeError("Can only take gradient of scalar-valued functions")
        else:
            end_node.add_outgrad(1.0)
            for node in tape[::-1]:
                node.send_upstream()
            return start_node.get_outgrad()

    return gradfun

def kyapply(fun, *args, **kwargs):
    tape = top_tape(args)
    if tape is None:
        return fun(*args, **kwargs)
    else:
        arg_vals = [arg.value if ismember(arg, tape) else arg for arg in args]
        result = kyapply(fun, *arg_vals, **kwargs)
        parent_ops = [(gradfuns[fun][i], parent)
                      for i, parent in enumerate(args) if ismember(parent, tape)]
        return new_node(result, tape, parent_ops, arg_vals, kwargs)

class CalculationTape(list):
    def __init__(self, prev_tape):
        super(CalculationTape, self).__init__([])
        self.priority = prev_tape.priority + 1 if prev_tape is not None else 1

def top_tape(args):
    tapes = [node.tape for node in args if isinstance(node, Node)]
    return max(tapes, key=attrgetter('priority')) if tapes else None

# ----- Nodes and subclasses for operator overloading -----

k = kyapply
getval = lambda x : x.raw_value if isinstance(x, Node) else x
ismember = lambda x, tape : isinstance(x, Node) and x.tape is tape
isarray  = lambda x : isinstance(getval(x), np.ndarray)
isscalar = lambda x : isinstance(getval(x), (float, np.float64))
isdict   = lambda x : isinstance(getval(x), dict)

class Node(object):
    __slots__ = ['value', 'raw_value', 'tape', 'parent_ops', 'args', 'kwargs', '_outgrad']
    def __init__(self, value, tape, parent_ops=[], args=(), kwargs=()):
        self.value = value
        self.raw_value = getval(value)
        tape.append(self)
        self.tape = tape
        self.args = args
        self.kwargs = kwargs
        self.parent_ops = parent_ops
        self._outgrad = None

    def send_upstream(self):
        if self._outgrad is None: return
        for gradfun, parent in self.parent_ops:
            parent.add_outgrad(gradfun(self._outgrad, *self.args, **self.kwargs))

    def add_outgrad(self, new):
        new = self.process_outgrad(new)
        if self._outgrad is None:
            self._outgrad = new
        else:
            self._outgrad = k(add_any, self._outgrad, new)

    def get_outgrad(self):
        if self._outgrad is None:
            self._outgrad = zeros_like(getval(self))
        return self._outgrad

    def __getitem__(self, idx):
        return k(take, self, idx)

    # Ensure precedence of Node's __rmul__ over numpy's __mul__
    __array_priority__ = 100.0

    # General operator overloads
    def __add__(self, other):  return k(op.add, self, other)
    def __radd__(self, other): return k(op.add, self, other)
    def __sub__(self, other):  return k(op.sub, self, other)
    def __rsub__(self, other): return k(op.sub, other, self)
    def __mul__(self, other):  return k(op.mul, self, other)
    def __rmul__(self, other): return k(op.mul, other, self)
    def __neg__(self):         return k(op.neg, self)
    def __pow__(self, power):  return k(op.pow, self, power)
    def __rpow__(self, power): return k(op.pow, power, self)
    def __div__(self, other):  return k(op.div, self, other)
    def __rdiv__(self, other): return k(op.div, other, self)
    def __lt__(self, other):   return getval(self) < getval(other)
    def __gt__(self, other):   return getval(self) > getval(other) 

class scalarNode(Node):
    def process_outgrad(self, new):
        return k(np.sum, new) if isarray(new) else new

class ndarrayNode(Node):
    def process_outgrad(self, new):
        # Handle broadcasting
        while new.ndim > self.ndim:
            new = k(np.sum, new, 0)
        for axis, size in enumerate(self.shape):
            if size is 1:
                new = k(np.sum, new, axis, keepdims=True)
        return new

    @property
    def T(self): return k(np.transpose, self)
    @property
    def shape(self): return self.raw_value.shape
    @property
    def ndim(self): return self.raw_value.ndim

class dictNode(Node):
    def process_outgrad(self, new):
        return new

class listNode(Node):
    pass

class numpy_wrapper_maker(object):
    # A bit of a hack, but this lets you use numpy functions in the
    # more familiar way instead of k(fun, *args).
    def __getattr__(self, function_name):
        return partial(k, getattr(np, function_name))
numpy_wrapper = numpy_wrapper_maker()

# ----- Easy gradients -----

gradfuns = {}
gradfuns[np.abs]  = lambda g, x : k(np.sign, x) * g
gradfuns[np.exp]  = lambda g, x : k(np.exp, x) * g
gradfuns[np.log]  = lambda g, x : g / x
gradfuns[np.sin]  = lambda g, x : g * k(np.cos, x)
gradfuns[np.cos]  = lambda g, x : - g * k(np.sin, x)
gradfuns[np.sign] = lambda g, x : 0.0
gradfuns[np.full] = [None, lambda g, shape, fill_value :  k(np.sum, g)]
gradfuns[np.expand_dims] = lambda g, x, axis : k(np.squeeze, g, axis)
gradfuns[np.squeeze]     = lambda g, x, axis : k(np.repeat,  g, x.shape[axis], axis)
gradfuns[np.repeat]      = lambda g, x, axis : k(np.sum, g, axis, keepdims=True)
gradfuns[np.transpose]   = lambda g, x : k(np.transpose, g)
gradfuns[op.neg] = [lambda g, x : - g]
gradfuns[op.add] = [lambda g, x, y : g,     lambda g, x, y : g]
gradfuns[op.mul] = [lambda g, x, y : y * g, lambda g, x, y : x * g]
gradfuns[op.sub] = [lambda g, x, y : g,     lambda g, x, y : - g]
gradfuns[op.div] = [lambda g, x, y : g / y, lambda g, x, y : - g * x / y**2]
gradfuns[op.pow] = [lambda g, x, y : g * y * x ** (y - 1),
                    lambda g, x, y : g * k(np.log, x) * x ** y]

# ----- Trickier ones -----

def grad_np_sum(g, x, axis=None, keepdims=False):
    if not isarray(x):
        return g
    if axis is None:
        return k(np.full, x.shape, g)
    elif not keepdims:
        g = k(np.expand_dims, g, axis)
    return k(np.repeat, g, x.shape[axis], axis)
gradfuns[np.sum] = grad_np_sum

def grad_np_dot_A(g, A, B):
    if B.ndim is 2:
        return k(np.dot, g, B.T)
    elif A.ndim is 2:
        return k(np.outer, g, B)
    else:
        return g * B
def grad_np_dot_B(g, A, B):
    if A.ndim is 2:
        return k(np.dot, A.T, g)
    elif B.ndim is 2:
        return k(np.outer, A, g)
    else:
        return g * A
gradfuns[np.dot] = [grad_np_dot_A, grad_np_dot_B]

# ----- New primitives and type handling -----

def take(A, idx):
    return A[idx]
def pad_zeros(x, example, idx):
    # This is very inefficient but introducing mutable types will take some thinking.
    A = zeros_like(example)
    A[idx] = x
    return A
gradfuns[take] = lambda g, x, idx : k(pad_zeros, g, getval(x), idx)
gradfuns[pad_zeros] = lambda g, x, example, idx : k(take, g, idx)

def new_node(value, *args):
    if isarray(value):
        return ndarrayNode(value, *args)
    elif isscalar(value):
        return scalarNode(value, *args)
    elif isdict(value):
        return dictNode(value, *args)
    else:
        raise TypeError("Cannot differentiate wrt data type {0}".format(type(value)))

def add_any(A, B):
    if isinstance(A, (np.ndarray, float, np.float64)):
        return A + B
    elif isinstance(A, list):
        return [add_any(a, b) for a, b in zip(A, B)]
    elif isinstance(A, dict):
        return {key : add_any(A[key], B[key]) for key in A}
    else:
        raise TypeError("Can't add type {0}".format(type(A)))
gradfuns[add_any] = [lambda g, A, B : g, lambda g, A, B : g]

def zeros_like(X):
    if isinstance(X, (float, np.float64)):
        return 0.0
    elif isinstance(X, np.ndarray):
        return np.zeros(X.shape)
    elif isinstance(X, dict):
        return {key : zeros_like(val) for key, val in X.iteritems()}
    elif isinstance(X, list):
        return [zeros_like(x) for x in X]
    else:
        raise TypeError("Can't make zeros like {0}".format(type(X)))

# ----- Process gradients -----

gradfuns = {key : val if isinstance(val, list) else [val]
            for key, val in gradfuns.iteritems()}
