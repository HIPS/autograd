"""Generates a graphviz DOT file of an evaluation trace.
Usage (need the dot binary, from the graphviz package, www.graphviz.org):

python2 dot_graph.py | dot -Tpdf -o graph.pdf
"""
from __future__ import print_function
import autograd.numpy as np
from autograd.tracer import trace, Node
from autograd import grad


class GraphNode(Node):
    # Records the full graph (could having this in tracer.py)
    def __init__(self, value, fun, args, kwargs, parent_argnums, parents):
        self.fun_name = fun.__name__
        self.args = args
        self.parents = dict(zip(parent_argnums, parents))
        self.isroot = False

    def initialize_root(self, x):
        self.isroot = True

    def __repr__(self):
        return 'node_{}'.format(id(self))

def trace_graph(f, x):
    start_node = GraphNode.new_root(x)
    _, node = trace(start_node, f, x)
    return node

dot_edge = '{} -> {} [color=gray30];\n'.format
dot_function_node = '{} [label="{}", shape=box, color=lightblue, style=filled];\n'.format
dot_variable_node = '{} [label="{}", color=orange, style=filled];\n'.format
dot_graph = 'digraph G {{{}}}'.format

def graph_to_dotfile(graph):
    visited = set()
    def node_to_fragment(node):
        visited.add(node)
        if node.isroot:
            return dot_variable_node(node, 'input')
        fragment = dot_function_node(node, node.fun_name)
        for argnum, arg in enumerate(node.args):
            if argnum in node.parents:
                parent = node.parents[argnum]
                fragment += dot_edge(parent, node)
                if parent not in visited:
                    fragment += node_to_fragment(parent)
            else:
                argnode = '{}_arg_{}'.format(node, argnum)
                fragment += dot_edge(argnode, node)
                fragment += dot_variable_node(argnode, arg)

        return fragment

    dot_body = node_to_fragment(graph)
    dot_body += dot_variable_node('output', 'output')
    dot_body += dot_edge(graph, 'output')
    return dot_graph(dot_body)

if __name__ == '__main__':
    def fun(x):
        y = np.sin(x)
        return (y + np.exp(x) - 0.5) * y

    print(graph_to_dotfile(trace_graph(fun, 1.0)))
