from __future__ import absolute_import
from autograd.core import primitive, Node, zeros_like_node, getval

class DictNode(Node):
    def zeros(self):
        return {k : zeros_like_node(v) for k, v in getval(self).iteritems()}
    def __getitem__(self, idx):
        return take(self, idx)
    def __iter__(self):
        return self.value.__iter__()
Node.add_subclass(DictNode, [dict])

class ListNode(Node):
    def zeros(self):
        return [zeros_like_node(v) for v in getval(self)]
    def __getitem__(self, idx):
        return take(self, idx)
    def __len__(self): return len(self.value)
Node.add_subclass(ListNode, [list])

def take(A, idx): return A[idx]
take = primitive(take, lambda ans, A, idx : [lambda g : untake(g, idx, lambda : zeros_like_node(A))])

def untake(x, idx, zeros_fun):
    result = zeros_fun()
    result[idx] = x
    return result
untake = primitive(untake, lambda ans, x, idx, zeros : [lambda g : take(g, idx)])
