from autograd.extend import defvjp
from .sparse_wrapper import dot

def dot_left_vjp(ans, sparse, dense):
    return lambda g: g * dot(sparse, dense)

def dot_right_vjp(ans, dense, sparse):
    return lambda g: g * dot(sparse, dense)


defvjp(dot, dot_left_vjp)
