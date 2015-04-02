import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad, quick_grad_check
from scipy.optimize import minimize

class WeightsParser(object):
    """A helper class to index into a parameter vector."""
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

def build_rnn(input_size, state_size, output_size):
    """Builds functions to compute the output of an RNN."""
    parser = WeightsParser()
    parser.add_shape('init_hiddens', (1, state_size))
    parser.add_shape('change',  (input_size + state_size + 1, state_size))
    parser.add_shape('predict', (state_size + 1, output_size))

    def update(input, hiddens, change_weights):
        return np.tanh(activations(change_weights, input, hiddens))

    def outputs(weights, inputs):
        """Goes from right to left, updating the state."""
        num_sequences = inputs.shape[1]
        hiddens = np.repeat(parser.get(weights, 'init_hiddens'), num_sequences, axis=0)
        change_weights  = parser.get(weights, 'change')
        predict_weights = parser.get(weights, 'predict')
        output = []
        for input in inputs:  # Iterate over time steps.
            hiddens = update(input, hiddens, change_weights)
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

def string_to_one_hot(string, maxchar):
    """Converts an ASCII string to a one-of-k encoding."""
    ascii = np.array([ord(c) for c in string]).T
    return np.array(ascii[:,None] == np.arange(maxchar)[None, :], dtype=int)

def one_hot_to_string(one_hot_matrix):
    return "".join([chr(np.argmax(c)) for c in one_hot_matrix])

def build_dataset(filename, sequence_length, alphabet_size, num_lines = -1, pad=""):
    """Loads a text file, and turns each line into an encoded sequence."""
    with open(filename) as f:
        content = f.readlines()
    content = content[:num_lines]
    content = [line for line in content if len(line) > 2]   # Remove blank lines
    seqs = np.zeros((sequence_length, len(content), alphabet_size))
    for ix, line in enumerate(content):
        padded_line = (pad + line + " " * sequence_length)[:sequence_length]
        seqs[:, ix, :] = string_to_one_hot(padded_line, alphabet_size)
    return seqs

if __name__ == '__main__':
    npr.seed(1)
    input_size = output_size = 128   # The first 128 ASCII characters are the common ones.
    state_size = 40
    seq_length = 30
    param_scale = 0.01
    train_iters = 100

    # Learn to predict our own source code.
    train_inputs   = build_dataset('rnn.py', seq_length, input_size, num_lines = 60, pad = " ")
    train_targets  = build_dataset('rnn.py', seq_length, input_size, num_lines = 60)

    pred_fun, loss_fun, frac_err, num_weights = build_rnn(input_size, state_size, output_size)

    def print_training_prediction(weights, train_inputs, train_targets):
        print "Training text                         Predicted text"
        logprobs = np.asarray(pred_fun(weights, train_inputs))
        for t in xrange(logprobs.shape[1]):
            training_text  = one_hot_to_string(train_targets[:,t,:])
            predicted_text = one_hot_to_string(logprobs[:,t,:])
            print training_text.replace('\n', ' ') + "| " + predicted_text.replace('\n', ' ')

    def callback(weights):
        print "Train loss:", loss_fun(weights, train_inputs, train_targets)
        print_training_prediction(weights, train_inputs, train_targets)

   # Build gradient of loss function using autograd.
    loss_and_grad = grad(loss_fun, return_function_value=True)

    # Wrap function to only have one argument, for scipy.minimize.
    def training_loss_and_grad(weights):
        return loss_and_grad(weights, train_inputs, train_targets)

    init_weights = npr.randn(num_weights) * param_scale
    # Check the gradients numerically, just to be safe
    quick_grad_check(loss_fun, init_weights, (train_inputs, train_targets))

    print "Training RNN..."
    result = minimize(training_loss_and_grad, init_weights, jac=True, method='CG',
                      options={'maxiter':train_iters}, callback=callback)
    trained_weights = result.x

    print
    print "Generating text from RNN..."
    num_letters = 30
    for t in xrange(20):
        text = " "
        for i in xrange(num_letters):
            seqs = string_to_one_hot(text, output_size)[:, np.newaxis, :]
            logprobs = pred_fun(trained_weights, seqs)[-1].ravel()
            text += chr(npr.choice(len(logprobs), p=np.exp(logprobs)))
        print text
