from __future__ import absolute_import
from autograd.core import primitive, Node, getval

class DictNode(Node):
    def __getitem__(self, idx):
        return take(self, idx)
    def __iter__(self):
        return self.value.__iter__()
Node.add_subclass(DictNode, [dict])

class ListNode(Node):
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
untake = primitive(untake, lambda ans, x, idx, zeros_fun : [lambda g : take(g, idx)])
