import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.tjp import jacobian
from autograd import jacobian as _jacobian

from itertools import product


def allclose(x, y): return x.shape == y.shape and np.allclose(x, y)

def test_dot():
    npr.seed(0)
    shapes = [(), (2,), (2, 2), (2, 2, 2)]
    array_pairs = [(npr.normal(size=s1), npr.normal(size=s2))
                for s1, s2 in product(shapes, shapes)]
    argnums = [0, 1]

    def check(A, B, argnum):
        res1 = jacobian(np.dot, argnum)(A, B)
        res2 = _jacobian(np.dot, argnum)(A, B)
        assert allclose(res1, res2)

    for A, B in array_pairs:
        for argnum in argnums:
            yield check, A, B, argnum
