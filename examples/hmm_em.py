from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd import grad
from functools import partial


def EM(init_params, data):
    def EM_update(params):
        natural_params = map(np.log, params)
        expected_stats = grad(log_partition_function)(natural_params, data)  # E step
        return map(normalize, expected_stats)                                # M step

    def fixed_point(f, x0):
        x1 = f(x0)
        while different(x0, x1):
            x0, x1 = x1, f(x1)
        return x1

    def different(params1, params2):
        return not all(map(np.allclose, params1, params2))

    return fixed_point(EM_update, init_params)


def normalize(a):
    def replace_zeros(a):
        return np.where(a > 0., a, 1.)
    return a / replace_zeros(a.sum(-1, keepdims=True))


def log_partition_function(natural_params, data):
    if isinstance(data, list):
        return sum(map(partial(log_partition_function, natural_params), data))

    log_pi, log_A, log_B = natural_params
    log_alpha = log_pi
    for y_t in data:
        log_alpha = logsumexp(log_alpha[:,None] + log_A, axis=0) + log_B[:,y_t]
    return logsumexp(log_alpha)


if __name__ == '__main__':
    np.random.seed(0)
    np.seterr(divide='ignore')

    data = np.array([
        0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,1,1,0,0,1,0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1])

    init_pi = normalize(npr.rand(3))
    init_A = normalize(npr.rand(3,3))
    init_B = normalize(npr.rand(3,2))
    init_params = (init_pi, init_A, init_B)

    pi, A, B = EM(init_params, data)
