import numpy as np
import numpy.random as npr
from scipy.optimize import fmin_cg
from autograd import grad

class WeightsParser(object):
    def __init__(self):
        self.idxs_and_shapes = {}
        self.N = 0

    def add_shape(self, name, shape):
        start = self.N
        self.N += np.prod(shape)
        self.idxs_and_shapes[name] = (slice(start, self.N), shape)

    def get(self, vect, name):
        idxs, shape = self.idxs_and_shapes[name]
        return np.reshape(vect[idxs], shape)

def activations(input, cell, hidden, weights):
    cat_state = np.concatenate((input, cell, hidden, np.ones((input.shape[0],1))), axis=1)
    return np.dot(cat_state, weights)

def sigmoid(x):
    return 0.5*(np.tanh(x) + 1.0)   # Output ranges from 0 to 1.

def logsumexp(X, axis=1):
    max_X = np.max(X)
    return max_X + np.log(np.sum(np.exp(X - max_X), axis=axis, keepdims=True))

def build_lstm(input_size, state_size, output_size):
    """Build functions to compute the output of an LSTM."""
    parser = WeightsParser()
    parser.add_shape('forget', (input_size + 2 * state_size + 1, state_size))
    parser.add_shape('change', (input_size + 2 * state_size + 1, state_size))
    parser.add_shape('ingate',   (input_size + 2 * state_size + 1, state_size))
    parser.add_shape('outgate', (input_size + 2 * state_size + 1, output_size))

    def update_lstm(input, hidden, cell, forget_weights, change_weights,
                                         ingate_weights, outgate_weights):
        """One iteration of an LSTM layer."""
        forget  = sigmoid(activations(input, cell, hidden, forget_weights))
        ingate  = sigmoid(activations(input, cell, hidden, ingate_weights))
        change  = np.tanh(activations(input, cell, hidden, change_weights))
        cell    = cell * forget + ingate * change
        outgate = sigmoid(activations(input, cell, hidden, outgate_weights))
        #hidden = outgate * np.tanh(cell)  # Forces state_size == output_size?
        hidden = outgate * cell  # Don't want to constrain logprobs, so no tanh.
        return hidden, cell

    def outputs(weights, inputs):
        """Goes from right to left, updating the state.
           Outputs normalized log-probabilities."""
        num_seqs = inputs.shape[1]
        hiddens = np.zeros((num_seqs, state_size))
        cells = np.zeros((num_seqs, output_size))
        output = np.zeros(inputs.shape)
        for t, cur_input in enumerate(inputs):  # Iterate over time steps.
            hiddens, state = update_lstm(cur_input, hiddens, cells,
                                         parser.get(weights, 'forget'),
                                         parser.get(weights, 'change'),
                                         parser.get(weights, 'ingate'),
                                         parser.get(weights, 'outgate'))
            output[t, :, :] = hiddens - logsumexp(hiddens)
        return output # Output normalized log-probabilities.

    def loss(weights, inputs, targets):
        logprobs = outputs(weights, inputs)
        return np.sum(logprobs * targets) / targets.shape[0]

    def frac_err(weights, inputs, targets):
        return np.mean(np.argmax(targets, axis=1) != np.argmax(outputs(weights, inputs), axis=1))

    return outputs, loss, frac_err, parser.N

def one_hot_ascii(x, K):
    ascii = np.array([ord(c) - 31 for c in x]).T
    return np.array(ascii[:,None] == np.arange(K)[None, :], dtype=int)

def build_dataset(filename, seq_length, seq_width, pad=""):
    with open(filename) as f:
        content = f.readlines()
    seqs = np.zeros((seq_length, len(content), seq_width))
    for ix, line in enumerate(content):
        padded_line = (pad + line).ljust(seq_length)
        seqs[:, ix, :] = one_hot_ascii(padded_line, seq_width)
    return seqs

def demo_lstm():
    npr.seed(1)
    input_size = 129 - 32
    output_size = state_size = input_size
    seq_length = 100
    param_scale = 0.1
    train_iters = 10

    train_inputs  = build_dataset('lstm.py'      ,  seq_length, input_size)
    train_targets = build_dataset('lstm.py'      ,  seq_length, input_size, pad = " ")
    test_inputs   = build_dataset('neural_net.py',  seq_length, input_size)
    test_targets  = build_dataset('neural_net.py',  seq_length, input_size, pad = " ")

    pred_fun, loss_fun, frac_err, num_weights = build_lstm(input_size, state_size, output_size)

    loss_grad = grad(loss_fun)   # Specifies gradient of loss function using autograd.

    def training_grad(weights):
        return loss_grad(weights, train_inputs, train_targets)
    def training_loss(weights):
        return loss_fun( weights, test_inputs, test_targets)

    init_weights = npr.randn(num_weights) * param_scale   # Initialize with random weights.
    print "Random error rate: ", frac_err(init_weights, test_inputs, test_targets)
    print "Random gradient: ",  np.sum(loss_grad(init_weights, train_inputs, train_targets))

    def callback(weights):
        print "Train loss: ", loss_fun(weights, train_inputs, train_targets), \
              "Test error: ", frac_err(weights, test_inputs,  test_targets)

    trained_weights = fmin_cg(training_loss, init_weights, fprime=training_grad,
                              maxiter=train_iters, callback=callback)
demo_lstm()