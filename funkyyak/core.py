import operator as op
import numpy as np

# ----- Autodiff logic -----

isnode = lambda x : isinstance(x, Node)
getval = lambda x : x.value if isinstance(x, Node) else x

def grad(fun, argnum=0):
    def gradfun(*args):
        args = list(args)
        tape = []
        args[argnum] = new_node(args[argnum], None, [], {}, tape)
        ans = fun(*args)
        if not isnode(ans): return 0.0
        ans.outgrad = 1.0
        for node in tape[::-1]:
            for i, arg in enumerate(node.args):
                if not isnode(arg): continue
                gradfun = gradfuns[node.fun][i]
                arg.outgrad += gradfun(node.outgrad, *map(getval, node.args),
                                       **node.kwargs)
        return node.outgrad

    return gradfun

def kyapply(fun, *args, **kwargs):
    parents = filter(isnode, args)
    if parents:
        value = kyapply(fun, *map(getval, args), **kwargs)
        return new_node(value, fun, args, kwargs, parents[0].tape)
    else:
        return fun(*args, **kwargs)

# ----- Nodes and subclasses for operator overloading -----

k = kyapply
isarrayish = lambda x : isinstance(x, (np.ndarray, numpyNode))

def new_node(value, *args):
    if isarrayish(value):
        return numpyNode(value, *args)
    else:
        return Node(value, *args)

class Node(object):
    __slots__ = ['fun', 'args', 'value', 'outgrad', 'kwargs', 'tape']
    def __init__(self, value, fun, args, kwargs, tape):
        self.fun = fun
        self.args = args
        self.value = value
        self.kwargs = kwargs
        self.tape = tape
        tape.append(self)
        self.outgrad = 0.0

    # Ensure precedence of Node's __rmul__ over numpy's __mul__
    __array_priority__ = 100.0

    def __add__(self, other):  return k(op.add, self, other)
    def __radd__(self, other): return k(op.add, self, other)
    def __sub__(self, other):  return k(op.sub, self, other)
    def __rsub__(self, other): return k(op.sub, other, self)
    def __mul__(self, other):  return k(op.mul, self, other)
    def __rmul__(self, other): return k(op.mul, other, self)
    def __neg__(self):         return k(op.neg, self)
    def __pow__(self, power):  return k(op.pow, self, power)
    def __div__(self, other):  return k(op.div, self, other)
    def __rdiv__(self, other): return k(op.div, other, self)
    def __lt__(self, other):   return self.value < getval(other)
    def __gt__(self, other):   return self.value > getval(other) 
    
class numpyNode(Node):
    def __init__(self, *args):
        super(numpyNode, self).__init__(*args)

    @property
    def T(self): return k(np.transpose, self)
    @property
    def shape(self): return self.value.shape
    @property
    def ndim(self): return self.value.ndim

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

gradfuns[op.add] = [lambda g, x, y : g, lambda g, x, y : g]
gradfuns[op.mul] = [lambda g, x, y : y * g, lambda g, x, y : x * g]
gradfuns[op.pow] =  lambda g, x, y : g * y * x ** (y - 1)
gradfuns[op.sub] = [lambda g, x, y : g, lambda g, x, y : - g]
gradfuns[op.neg] = [lambda g, x : - g]
gradfuns[op.div] = [lambda g, x, y : g / y, lambda g, x, y : - g * x / y**2]

# ----- Trickier ones -----

def grad_np_sum(g, x, axis=None, keepdims=False):
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

for fun in [op.add, op.mul, op.sub, op.div]:
    for i, gradfun in enumerate(gradfuns[fun]):
        gradfuns[fun][i] = make_unbroadcasting(gradfun, i)
