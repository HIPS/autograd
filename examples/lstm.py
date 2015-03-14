import numpy as np
import numpy.random as npr
from scipy.optimize import fmin_cg
from autograd import grad

class WeightsParser(object):
    """A helper class to index into a combined parameter vector."""
    def __init__(self):
        self.idxs_and_shapes = {}
        self.num_weights = 0

    def add_shape(self, name, shape):
        start = self.num_weights
        self.num_weights += np.prod(shape)
        self.idxs_and_shapes[name] = (slice(start, self.num_weights), shape)

    def get(self, vect, name):
        idxs, shape = self.idxs_and_shapes[name]
        return np.reshape(vect[idxs], shape)

def sigmoid(x):
    return 0.5*(np.tanh(x) + 1.0)   # Output ranges from 0 to 1.

def activations(weights, *args):
    cat_state = np.concatenate(args + (np.ones((args[0].shape[0],1)),), axis=1)
    return np.dot(cat_state, weights)

def logsumexp(X, axis=1):
    max_X = np.max(X)
    return max_X + np.log(np.sum(np.exp(X - max_X), axis=axis, keepdims=True))

def build_lstm(input_size, state_size, output_size):
    """Builds functions to compute the output of an LSTM."""
    parser = WeightsParser()
    parser.add_shape('init_cells',   (1, state_size))
    parser.add_shape('init_hiddens', (1, state_size))
    parser.add_shape('forget',  (input_size + 2 * state_size + 1, state_size))
    parser.add_shape('change',  (input_size + 2 * state_size + 1, state_size))
    parser.add_shape('ingate',  (input_size + 2 * state_size + 1, state_size))
    parser.add_shape('outgate', (input_size + 2 * state_size + 1, state_size))
    parser.add_shape('predict', (state_size + 1, output_size))

    def update_lstm(input, hiddens, cells, forget_weights, change_weights,
                                           ingate_weights, outgate_weights):
        """One iteration of an LSTM layer."""
        change  = np.tanh(activations(change_weights, input, cells, hiddens))
        forget  = sigmoid(activations(forget_weights, input, cells, hiddens))
        ingate  = sigmoid(activations(ingate_weights, input, cells, hiddens))
        cells   = cells * forget + ingate * change
        outgate = sigmoid(activations(outgate_weights, input, cells, hiddens))
        hiddens = outgate * np.tanh(cells)
        return hiddens, cells

    def outputs(weights, inputs):
        """Goes from right to left, updating the state."""
        num_sequences = inputs.shape[1]
        hiddens = np.repeat(parser.get(weights, 'init_hiddens'), num_sequences, axis=0)
        cells   = np.repeat(parser.get(weights, 'init_cells'),   num_sequences, axis=0)
        forget_weights  = parser.get(weights, 'forget')
        change_weights  = parser.get(weights, 'change')
        ingate_weights  = parser.get(weights, 'ingate')
        outgate_weights = parser.get(weights, 'outgate')
        predict_weights = parser.get(weights, 'predict')
        output = []
        for input in inputs:  # Iterate over time steps.
            hiddens, cells = update_lstm(input, hiddens, cells, forget_weights,
                                         change_weights, ingate_weights, outgate_weights)
            cur_output = activations(predict_weights, hiddens)
            output.append(cur_output - logsumexp(cur_output))
        return output # Output normalized log-probabilities.

    def loss(weights, inputs, targets):
        logprobs = outputs(weights, inputs)
        loss_sum = 0.0
        for t in xrange(len(targets)):  # For every time step
            loss_sum -= np.sum(logprobs[t] * targets[t])
        return loss_sum / targets.shape[0] / targets.shape[1]

    def frac_err(weights, inputs, targets):
        return np.mean(np.argmax(targets, axis=2) != np.argmax(outputs(weights, inputs), axis=2))

    return outputs, loss, frac_err, parser.num_weights

def one_hot_ascii(string, maxchar):
    ascii = np.array([ord(c) for c in string]).T
    return np.array(ascii[:,None] == np.arange(maxchar)[None, :], dtype=int)

def build_dataset(filename, sequence_length, alphabet_size, lines = -1, pad=""):
    with open(filename) as f:
        content = f.readlines()
    content = content[:lines]
    content = [line for line in content if len(line) > 2]   # Remove blank lines
    seqs = np.zeros((sequence_length, len(content), alphabet_size))
    for ix, line in enumerate(content):
        padded_line = (pad + line + " " * sequence_length)[:sequence_length]
        seqs[:, ix, :] = one_hot_ascii(padded_line, alphabet_size)
    return seqs

def demo_lstm():
    npr.seed(1)
    input_size = output_size = 128   # The first 128 ASCII characters are the common ones.
    state_size = 4
    seq_length = 30
    param_scale = 0.01
    train_iters = 150

    train_inputs   = build_dataset('lstm.py', seq_length, input_size, lines = 2, pad = " ")
    train_targets  = build_dataset('lstm.py', seq_length, input_size, lines = 2)

    pred_fun, loss_fun, frac_err, num_weights = build_lstm(input_size, state_size, output_size)

    loss_grad = grad(loss_fun)   # Specifies gradient of loss function using autograd.

    def training_grad(weights):
        return loss_grad(weights, train_inputs, train_targets)
    def training_loss(weights):
        return loss_fun( weights, train_inputs, train_targets)

    def print_training_prediction(weights, train_inputs, train_targets):
        print "Training text                         Predicted text"
        logprobs = np.asarray(pred_fun(weights, train_inputs))
        for t in xrange(logprobs.shape[1]):
            training_text  = "".join([chr(np.argmax(c)) for c in train_targets[:,t,:]])
            predicted_text = "".join([chr(np.argmax(c)) for c in logprobs[:,t,:]])
            print training_text.replace('\n','') + "| " + predicted_text.replace('\n','')

    def callback(weights):
        print "Train loss:", loss_fun(weights, train_inputs, train_targets)
        print_training_prediction(weights, train_inputs, train_targets)

    print "Training LSTM model..."
    weights = npr.randn(num_weights) * param_scale
    weights = fmin_cg(training_loss, weights, fprime=training_grad,
                      maxiter=train_iters, callback=callback)

    print "Generating text from LSTM model..."
    num_letters = 30
    for t in xrange(20):
        text = " "
        for i in xrange(num_letters):
            seqs = one_hot_ascii(text, output_size)[:, np.newaxis, :]
            logprobs = pred_fun(weights, seqs)[-1].ravel()
            text += chr(npr.choice(len(logprobs), p=np.exp(logprobs)))
        print text

demo_lstm()