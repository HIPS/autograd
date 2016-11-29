"""
Module containing reverse mode functionality.
"""
from autograd.core import (Node, FloatNode, tape_computation, zeros_like,
                           new_node, getval, attach_name_and_doc,
                           cast_to_node_type, NoDerivativeNode)
import warnings


class ReverseModeTape(list):
    """
    This tape is fundamentally a list of 'ReverseNode' objects which store the
    information necessary to run the backward pass.
    """
    def __init__(self):
        self.recording = False
        # Mapping from each forward node to the corresponding reverse node and
        # each reverse node to the corresponding forward node.
        self.node_mappings = {}

    def start_recording(self, start_node):
        first_reverse_node = ReverseNode()
        self.node_mappings[start_node] = first_reverse_node
        self.node_mappings[first_reverse_node] = start_node
        self.append(first_reverse_node)
        self.recording = True

    def stop_recording(self):
        self.recording = False

    def carry_forward(self, primitive, argnum):
        # If any of the gradients are not zero then we need to carry this
        # tape forward.
        return self.recording and argnum not in primitive.zero_grads

    def record_new_node(self, node):
        if isinstance(node, NoDerivativeNode):
            reverse_node = NoDerivativeReverseNode()
        else:
            reverse_node = ReverseNode()
        self.node_mappings[node] = reverse_node
        self.node_mappings[reverse_node] = node

    def update(self, primitive, args, kwargs, result):
        operations = []

        for i, arg in enumerate(args):
            if isinstance(arg, Node) and self in arg.tapes:
                if i in primitive.zero_grads:
                    continue
                parent_reverse_node = self.node_mappings[arg]
                gradfun = primitive.gradmaker(i, result, args, kwargs)
                operations.append((gradfun, parent_reverse_node))

        if operations:
            reverse_node = self.node_mappings[result]
            reverse_node.operations = operations
            self.append(reverse_node)

    def __hash__(self):
        return id(self)


class ReverseNode(object):
    """
    This stores the information necessary to do the backward pass.
    """
    __slots__ = ['outgrads', 'operations']

    def __init__(self, operations=[]):
        self.outgrads = []

        # A list of 2-tuples of the form (gradfun, parent_reverse_node) where
        # gradfun is a function of the outgrad.
        self.operations = operations

    def __hash__(self):
        return id(self)


class NoDerivativeReverseNode(ReverseNode):
    def __init__(self, operations=[]):
        super(NoDerivativeReverseNode, self).__init__(operations)


def grad(fun, argnum=0):
    """Returns a function which computes the gradient of `fun` with respect to
    positional argument number `argnum`. The returned function takes the same
    arguments as `fun`, but returns the gradient instead. The function `fun`
    should be scalar-valued. The gradient has the same type as the argument."""
    tape = ReverseModeTape()

    @attach_name_and_doc(fun, argnum, 'Gradient')
    def gradfun(*args, **kwargs):
        return backward_pass(*tape_computation(fun, args, kwargs, tape,
                                               argnum))
    return gradfun


def backward_pass(start_node, end_node, tape, preserve_tape=False):
    if not isinstance(end_node, Node) or tape not in end_node.tapes:
        warnings.warn("Output seems independent of input. Returning zero "
                      "gradient.")
        return zeros_like(start_node)

    # Check that the end node can be cast to a float.
    if type(end_node) is not FloatNode:
        try:
            FloatNode.cast(end_node, 1.0)
        except TypeError:
            raise TypeError(
                "Output type {} can't be cast to float. Function grad "
                "requires a scalar-valued function. Try jacobian or "
                "elementwise_grad.".format(type(end_node.value)))

    tape.node_mappings[end_node].outgrads = [cast_to_node_type(1.0,
                                                               type(end_node),
                                                               end_node.value)]

    if preserve_tape:
        position = len(tape) - 1
    while tape:
        # If we're preserving the tape then we want this while loop to act as
        # a reverse for loop. There may be a nicer way to achieve this.
        if preserve_tape:
            if position == -1:
                break
            reverse_node = tape[position]
            position -= 1
        else:
            reverse_node = tape.pop()

        if reverse_node.outgrads:
            node = tape.node_mappings[reverse_node]
            if isinstance(reverse_node, NoDerivativeReverseNode):
                raise TypeError("Can't differentiate wrt {0}".
                                format(type(node.value)))
            current_outgrad = (node.sum_grads(reverse_node.outgrads))
            reverse_node.outgrads = []
            assert type(new_node(getval(current_outgrad))) == type(node), \
                "Types are {0} and {1}".format(
                    type(new_node(getval(current_outgrad))), type(node))
            for gradfun, parent in reverse_node.operations:
                parent_type = type(tape.node_mappings[parent])
                parent_value = tape.node_mappings[parent].value
                parent_outgrad = cast_to_node_type(gradfun(current_outgrad),
                                                   parent_type, parent_value)
                parent.outgrads.append(parent_outgrad)

    return current_outgrad
