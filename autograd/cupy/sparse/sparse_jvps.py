from autograd.extend import def_linear
from .sparse_wrapper import dot

def_linear(dot)
