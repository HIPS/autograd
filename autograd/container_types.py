from __future__ import absolute_import
from autograd.core import Node, zeros_like, getval

class DictNode(Node):
    def zeros(self):
        return {k : zeros_like(v) for k, v in getval(self).iteritems()}

    def __iter__(self):
        return self.value.__iter__()
Node.add_subclass(DictNode, [dict])

class ListNode(Node):
    def zeros(self):
        return [zeros_like(v) for v in getval(self)]

    def __len__(self): return len(self.value)
Node.add_subclass(ListNode, [list])
