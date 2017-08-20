from . import numpy_wrapper as anp
from autograd.core import defjvp, defjvps, defjvp_is_zero, defjvp_is_identical

# TODO: broadcast these binary operators
defjvp(anp.add, lambda g, ans, x, y : g)
defjvp(anp.add, lambda g, ans, x, y : g, argnum=1)
defjvp_is_identical(anp.multiply, argnum=0)
defjvp_is_identical(anp.multiply, argnum=1)
defjvp(anp.sin, lambda g, ans, x : g * anp.cos(x))
defjvp(anp.power, lambda g, ans, x, y : g * y * x ** (y-1))
