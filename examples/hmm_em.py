from __future__ import division, print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd.convenience_wrappers import value_and_grad as vgrad
from functools import partial
from os.path import join, dirname
import string


def EM(init_params, data, callback=None):
    def EM_update(params):
        natural_params = list(map(np.log, params))
        loglike, E_stats = vgrad(log_partition_function)(natural_params, data)  # E step
        if callback: callback(loglike, params)
        return list(map(normalize, E_stats))                                    # M step

    def fixed_point(f, x0):
        x1 = f(x0)
        while different(x0, x1):
            x0, x1 = x1, f(x1)
        return x1

    def different(params1, params2):
        allclose = partial(np.allclose, atol=1e-3, rtol=1e-3)
        return not all(map(allclose, params1, params2))

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
    for y in data:
        log_alpha = logsumexp(log_alpha[:,None] + log_A, axis=0) + log_B[:,y]

    return logsumexp(log_alpha)


def initialize_hmm_parameters(num_states, num_outputs):
    init_pi = normalize(npr.rand(num_states))
    init_A = normalize(npr.rand(num_states, num_states))
    init_B = normalize(npr.rand(num_states, num_outputs))
    return init_pi, init_A, init_B


def build_dataset(filename, max_lines=-1):
    """Loads a text file, and turns each line into an encoded sequence."""
    encodings = dict(list(map(reversed, enumerate(string.printable))))
    digitize = lambda char: encodings[char] if char in encodings else len(encodings)
    encode_line = lambda line: np.array(list(map(digitize, line)))
    nonblank_line = lambda line: len(line) > 2

    with open(filename) as f:
        lines = f.readlines()

    encoded_lines = list(map(encode_line, list(filter(nonblank_line, lines))[:max_lines]))
    num_outputs = len(encodings) + 1

    return encoded_lines, num_outputs


if __name__ == '__main__':
    np.random.seed(0)
    np.seterr(divide='ignore')

    # callback to print log likelihoods during training
    print_loglike = lambda loglike, params: print(loglike)

    # load training data
    lstm_filename = join(dirname(__file__), 'lstm.py')
    train_inputs, num_outputs = build_dataset(lstm_filename, max_lines=60)

    # train with EM
    num_states = 20
    init_params = initialize_hmm_parameters(num_states, num_outputs)
    pi, A, B = EM(init_params, train_inputs, print_loglike)
