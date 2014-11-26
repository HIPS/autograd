import operator as op
import numpy as np
from functools import partial

# ----- Autodiff logic -----

def grad(fun, argnum=0):
    def gradfun(*args):
        args = list(args)
        tape = CalculationTape(highest_tape(args))
        start_node = new_node(args[argnum], tape)
        args[argnum] = start_node
        ans = fun(*args)
        if not isnode(ans): return 0.0
        ans.outgrad = 1.0
        for step_back in tape[::-1]:
            step_back()
        return start_node.outgrad

    return gradfun

def kyapply(fun, *args, **kwargs):
    tape = highest_tape(args)
    if tape is not None:
        is_parent = lambda x : isnode(x) and x.tape is tape
        arg_vals = [arg.value if is_parent(arg) else arg for arg in args]
        node = new_node(kyapply(fun, *arg_vals, **kwargs), tape)
        for i, arg in enumerate(args):
            if not is_parent(arg): continue
            tape.append(partial(send_grad_back, node, gradfuns[fun][i],
                                arg, arg_vals, kwargs))
        return node
    else:
        return fun(*args, **kwargs)

def send_grad_back(node, gradfun, parent, args, kwargs):
    parent.outgrad += gradfun(node.outgrad, *args, **kwargs)

class CalculationTape(list):
    def __init__(self, prev_tape):
        super(CalculationTape, self).__init__([])
        self.priority = 1 if prev_tape is None else prev_tape.priority + 1

def highest_tape(args):
    tapes = [node.tape for node in filter(isnode, args)]
    return max(tapes, key=lambda x : x.priority) if tapes else None

isnode = lambda x : isinstance(x, Node)
getval = lambda x : getval(x.value) if isnode(x) else x

# ----- Nodes and subclasses for operator overloading -----

k = kyapply
isarrayish = lambda x : isinstance(x, (np.ndarray, numpyNode))

def new_node(value, tape):
    if isarrayish(value):
        return numpyNode(value, tape)
    else:
        return Node(value, tape)

class Node(object):
    __slots__ = ['value', 'tape', 'outgrad']
    def __init__(self, value, tape):
        self.tape = tape
        self.value = value
        self.outgrad = 0.0

    # Ensure precedence of Node's __rmul__ over numpy's __mul__
    __array_priority__ = 100.0

    # Numpy overloads. A better mechanism, __numpy_ufunc__, is expected in numpy v1.10
    def dot(self, other): return k(np.dot, self, other)
    def sum(self, axis=None, **kwargs) : return k(np.sum, self, axis=axis)
    def exp(self): return k(np.exp, self)
    def log(self): return k(np.log, self)
    def sin(self): return k(np.sin, self)
    def cos(self): return k(np.cos, self)

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
    
class numpyNode(Node):
    def __init__(self, *args):
        super(numpyNode, self).__init__(*args)

    @property
    def T(self): return k(np.transpose, self)
    @property
    def shape(self): return self.value.shape
    @property
    def ndim(self): return self.value.ndim

class dictNode(Node):
    pass

class listNode(Node):
    pass

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
    if not isarrayish(x):
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

# ----- Process gradients -----

def make_unbroadcasting(fun, i):
    def unbroadcasting_fun(g, *args):
        new_x = fun(g, *args)
        old_x = args[i]
        if isarrayish(new_x) and isarrayish(old_x):
            while new_x.ndim > old_x.ndim:
                new_x = k(np.sum, new_x, 0)
            for axis, size in enumerate(old_x.shape):
                if size is 1:
                    new_x = k(np.sum, new_x, axis, keepdims=True)
        elif isarrayish(new_x):
            new_x = k(np.sum, new_x)
        return new_x

    return unbroadcasting_fun

gradfuns = {k : v if isinstance(v, list) else [v]
            for k, v in gradfuns.iteritems()}

for fun in [op.add, op.mul, op.sub, op.div, op.pow]:
    for i, gradfun in enumerate(gradfuns[fun]):
        gradfuns[fun][i] = make_unbroadcasting(gradfun, i)
