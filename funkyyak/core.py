import operator as op
import numpy as np

isnode = lambda x : isinstance(x, Node)
getval = lambda x : x.value if isinstance(x, Node) else x

def grad(fun, argnum=0):
    def gradfun(*args):
        args = list(args)
        tape = []
        args[argnum] = Node(None, [], args[argnum], tape)
        ans = fun(*args)
        if not isnode(ans): return 0.0
        ans.outgrad = 1.0
        for node in tape[::-1]:
            for i, arg in enumerate(node.args):
                if not isnode(arg): continue
                gradfun = gradfuns[node.fun][i]
                arg.outgrad += gradfun(node.outgrad, *map(getval, node.args))
        return node.outgrad

    return gradfun

def kyapply(fun, *args):
    parents = filter(isnode, args)
    if parents:
        value = kyapply(fun, *map(getval, args))
        return Node(fun, args, value, parents[0].tape)
    else:
        return fun(*args)

class Node(object):
    __slots__ = ['fun', 'args', 'value', 'outgrad', 'tape']
    def __init__(self, fun, args, value, tape):
        self.fun = fun
        self.args = args
        self.value = value
        self.tape = tape
        tape.append(self)
        self.outgrad = 0.0

    @property
    def T(self):
        return k(np.transpose, self)

    def __add__(self, other):  return k(op.add, self, other)
    def __radd__(self, other): return k(op.add, self, other)
    def __sub__(self, other):  return k(op.sub, self, other)
    def __rsub__(self, other): return k(op.sub, other, self)
    def __mul__(self, other):  return k(op.mul, self, other)
    def __rmul__(self, other): return k(op.mul, self, other)
    def __neg__(self):         return k(op.neg, self)
    def __pow__(self, power):  return k(op.pow, self, power)
    def __div__(self, other):  return k(op.div, self, other)
    def __lt__(self, other):   return self.value < getval(other)
    def __gt__(self, other):   return self.value > getval(other) 
    
k = kyapply
gradfuns = {}

# ----- Numpy gradients -----

gradfuns[np.dot]  = [lambda g, A, B : k(np.dot, g.T, B),
                     lambda g, A, B : k(np.dot, A, g.T)]
gradfuns[np.sign] = [lambda g, x : 0.0]
gradfuns[np.abs]  = [lambda g, x : k(np.sign, x) * g]
gradfuns[np.exp]  = [lambda g, x : k(np.exp, x) * g]
gradfuns[np.log]  = [lambda g, x : g / x]
gradfuns[np.sin]  = [lambda g, x : g * k(np.cos, x)]
gradfuns[np.cos]  = [lambda g, x : - g * k(np.sin, x)]
gradfuns[np.transpose] = [lambda g, x : k(np.transpose, g)]

# ----- Operator gradients -----

gradfuns[op.add] = [lambda g, x, y : g, lambda g, x, y : g]
gradfuns[op.mul] = [lambda g, x, y : y * g, lambda g, x, y : x * g]
gradfuns[op.pow] = [lambda g, x, y : g * y * x ** (y - 1), None]
gradfuns[op.sub] = [lambda g, x, y : g, lambda g, x, y : - g]
gradfuns[op.neg] = [lambda g, x : - g]
gradfuns[op.div] = [lambda g, x, y : g / y, lambda g, x, y : - x / y**2]
