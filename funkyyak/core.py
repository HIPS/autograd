import operator as op
import numpy as np
import itertools as it
from functools import partial
from operator import attrgetter
from collections import namedtuple

# ----- Autodiff logic -----

def grad(fun, argnum=0):
    def gradfun(*args):
        tape = CalculationTape(top_tape(args))
        start_node = Node(args[argnum], tape)
        args = args[:argnum] + (start_node,) + args[argnum+1:]
        end_node = fun(*args)
        if not tape.hasmember(end_node):
            return start_node.sum_outgrads()
        if not isfloat(end_node):
            raise TypeError("Can only take gradient of scalar-valued functions")
        else:
            end_node.outgrads.append(1.0)
            for node in tape[::-1]:
                node.send_upstream()
            return start_node.sum_outgrads()

    return gradfun

def kyapply(fun, *args, **kwargs):
    tape = top_tape(args)
    if tape is None:
        return fun(*args, **kwargs)
    else:
        arg_vals = [arg.value if tape.hasmember(arg) else arg for arg in args]
        result = kyapply(fun, *arg_vals, **kwargs)
        parent_ops = [(gradfuns[fun][i], parent)
                      for i, parent in enumerate(args) if tape.hasmember(parent)
                      and gradfuns[fun][i] is not None]
        return Node(result, tape, parent_ops, arg_vals, kwargs, fun)
k = kyapply

class CalculationTape(list):
    def __init__(self, prev_tape):
        super(CalculationTape, self).__init__([])
        self.priority = prev_tape.priority + 1 if prev_tape is not None else 1

    def hasmember(self, x):
        return isinstance(x, Node) and x.tape is self

def top_tape(args):
    tapes = [node.tape for node in args if isinstance(node, Node)]
    return max(tapes, key=attrgetter('priority')) if tapes else None

class Node(object):
    __slots__ = ['value', 'tape', 'parent_ops', 'args', 'kwargs', 'outgrads', 'fun']
    def __init__(self, value, tape, parent_ops=[], args=(), kwargs={}, fun=None):
        if not isinstance(value, (Node, float, np.ndarray, dict, Setter)):
            raise TypeError("Can't differentiate wrt {0}".format(type(value)))
        self.value = value
        self.tape = tape
        tape.append(self)
        self.args = args
        self.kwargs = kwargs
        self.parent_ops = parent_ops
        self.outgrads = []
        self.fun = fun

    def send_upstream(self):
        if self.outgrads:
            outgrad_sum = self.sum_outgrads()
            args, kwargs = self.args, self.kwargs
            for gradfun, parent in self.parent_ops:
                parent.outgrads.append(gradfun(outgrad_sum, *args, **kwargs))

    def sum_outgrads(self):
        if len(self.outgrads) is 1 and not issetter(self.outgrads[0]):
            return self.outgrads[0]
        else:
            outgrad_sum = zeros_like(getval(self))
            for new in self.outgrads:
                outgrad_sum = k(mutating_add, outgrad_sum, new)
            return outgrad_sum

    # Ensure precedence of Node's __rmul__ over numpy's __mul__
    __array_priority__ = 100.0

    # Operator overloads and familiar methods
    @property
    def T(self): return k(np.transpose, self)
    @property
    def shape(self): return self.value.shape
    @property
    def ndim(self): return self.value.ndim

    def __getitem__(self, idx): return k(take, self, idx)
    def __add__(self, other):   return k(op.add, self, other)
    def __radd__(self, other):  return k(op.add, self, other)
    def __sub__(self, other):   return k(op.sub, self, other)
    def __rsub__(self, other):  return k(op.sub, other, self)
    def __mul__(self, other):   return k(op.mul, self, other)
    def __rmul__(self, other):  return k(op.mul, other, self)
    def __neg__(self):          return k(op.neg, self)
    def __pow__(self, power):   return k(op.pow, self, power)
    def __rpow__(self, power):  return k(op.pow, power, self)
    def __div__(self, other):   return k(op.div, self, other)
    def __rdiv__(self, other):  return k(op.div, other, self)
    def __lt__(self, other):    return getval(self) < getval(other)
    def __gt__(self, other):    return getval(self) > getval(other) 

# ----- Helper functions -----

def getval(x)   : return getval(x.value) if isinstance(x, Node) else x
def isarray(x)  : return isinstance(getval(x), np.ndarray)
def isfloat(x)  : return isinstance(getval(x), float)
def issetter(x) : return isinstance(getval(x), Setter)

def zeros_like(x):
    if isinstance(x, float):
        return 0.0
    elif isinstance(x, np.ndarray):
        return np.zeros(x.shape)
    elif isinstance(x, dict):
        return {k : zeros_like(v) for k, v in x.iteritems()}
    else:
        raise TypeError("Can't produce zeros like {0}".format(type(x)))

class numpy_wrapper_maker(object):
    # A bit of a hack, but this lets you use numpy functions in the
    # more familiar way instead of k(fun, *args).
    def __getattr__(self, function_name):
        return partial(k, getattr(np, function_name))
numpy_wrapper = numpy_wrapper_maker()

# ----- Easy gradients -----

gradfuns = {}
gradfuns[np.abs]  = [lambda g, x : k(np.sign, x) * g]
gradfuns[np.exp]  = [lambda g, x : k(np.exp, x) * g]
gradfuns[np.log]  = [lambda g, x : g / x]
gradfuns[np.sin]  = [lambda g, x : g * k(np.cos, x)]
gradfuns[np.cos]  = [lambda g, x : - g * k(np.sin, x)]
gradfuns[np.tan]  = [lambda g, x : g / k(np.cos, x) **2]
gradfuns[np.sinh] = [lambda g, x : g * k(np.cosh, x)]
gradfuns[np.cosh] = [lambda g, x : g * k(np.sinh, x)]
gradfuns[np.tanh] = [lambda g, x : g / k(np.cosh, x) **2]
gradfuns[np.sign] = [None]
gradfuns[np.full] = [None, lambda g, shape, fill_value :  k(np.sum, g)]
gradfuns[np.reshape]     = [lambda g, x, shape: k(np.reshape, g, x.shape)] 
gradfuns[np.expand_dims] = [lambda g, x, axis : k(np.squeeze, g, axis)]
gradfuns[np.squeeze]     = [lambda g, x, axis : k(np.repeat,  g, x.shape[axis], axis)]
gradfuns[np.repeat]      = [lambda g, x, shape, axis : k(np.sum, g, axis, keepdims=True)]
gradfuns[np.transpose]   = [lambda g, x : k(np.transpose, g)]
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
gradfuns[np.sum] = [grad_np_sum]

def grad_np_max(g, x):
    idxs = np.argmax(getval(x))
    return k(untake, g, np.unravel_index(idxs, x.shape))
gradfuns[np.max] = [grad_np_max]

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

# ----- New primitives -----

Setter = namedtuple('Setter', ('idx', 'val'))

def mutating_add(old, new):
    if isinstance(new, Setter):
        old[new.idx] += new.val
    else:
        old += new
    return old
gradfuns[mutating_add] = [lambda g, old, new : g] * 2

def take(A, idx):   return A[idx]
def untake(x, idx): return Setter(idx, x)
gradfuns[take]   = [lambda g, x, idx : k(untake, g, idx)]
gradfuns[untake] = [lambda g, x, idx : k(take, g, idx)]

# ----- Process gradients -----

def undo_broadcast(fun, argnum):
    def new_fun(g, *args):
        ans = fun(g, *args)
        x = args[argnum]
        if isfloat(x) and isarray(ans):
            ans = k(np.sum, ans)
        elif isarray(x):
            while ans.ndim > x.ndim:
                ans = k(np.sum, ans, axis=0)
            for axis, size in enumerate(x.shape):
                if size is 1:
                    ans = k(np.sum, ans, axis, keepdims=True)
        return ans

    return new_fun

broadcasting_ops = [op.add, op.mul, op.sub, op.div, op.pow]
for fun, argnum in it.product(broadcasting_ops, [0, 1]):
    gradfuns[fun][argnum] = undo_broadcast(gradfuns[fun][argnum], argnum)
