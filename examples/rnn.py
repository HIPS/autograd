"""Implements the long-short term memory character model.
This version vectorizes over multiple examples, but each string
has a fixed length."""

from __future__ import absolute_import
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.scipy.misc import logsumexp
from os.path import dirname, join
from builtins import range
from autograd.optimizers import adam


### Helper functions #################

def sigmoid(x):
    return 0.5*(np.tanh(x) + 1.0)   # Output ranges from 0 to 1.

def concat_and_multiply(weights, *args):
    cat_state = np.hstack(args + (np.ones((args[0].shape[0], 1)),))
    return np.dot(cat_state, weights)


### Define recurrent neural net #######

def create_rnn_params(input_size, state_size, output_size,
                      param_scale=0.01, rs=npr.RandomState(0)):
    return {'init hiddens': rs.randn(1, state_size) * param_scale,
            'change':       rs.randn(input_size + state_size + 1, state_size) * param_scale,
            'predict':      rs.randn(state_size + 1, output_size) * param_scale}

def rnn_predict(params, inputs):
    def update_rnn(input, hiddens):
        return np.tanh(concat_and_multiply(params['change'], input, hiddens))

    def hiddens_to_output_probs(hiddens):
        output = concat_and_multiply(params['predict'], hiddens)
        return output - logsumexp(output, axis=1, keepdims=True)     # Normalize log-probs.

    num_sequences = inputs.shape[1]
    hiddens = np.repeat(params['init hiddens'], num_sequences, axis=0)
    output = [hiddens_to_output_probs(hiddens)]

    for input in inputs:  # Iterate over time steps.
        hiddens = update_rnn(input, hiddens)
        output.append(hiddens_to_output_probs(hiddens))
    return output

def rnn_log_likelihood(params, inputs, targets):
    logprobs = rnn_predict(params, inputs)
    loglik = 0.0
    num_time_steps, num_examples, _ = inputs.shape
    for t in range(num_time_steps):
        loglik += np.sum(logprobs[t] * targets[t])
    return loglik / (num_time_steps * num_examples)


### Dataset setup ##################

def string_to_one_hot(string, maxchar):
    """Converts an ASCII string to a one-of-k encoding."""
    ascii = np.array([ord(c) for c in string]).T
    return np.array(ascii[:,None] == np.arange(maxchar)[None, :], dtype=int)

def one_hot_to_string(one_hot_matrix):
    return "".join([chr(np.argmax(c)) for c in one_hot_matrix])

def build_dataset(filename, sequence_length, alphabet_size, max_lines=-1):
    """Loads a text file, and turns each line into an encoded sequence."""
    with open(filename) as f:
        content = f.readlines()
    content = content[:max_lines]
    content = [line for line in content if len(line) > 2]   # Remove blank lines
    seqs = np.zeros((sequence_length, len(content), alphabet_size))
    for ix, line in enumerate(content):
        padded_line = (line + " " * sequence_length)[:sequence_length]
        seqs[:, ix, :] = string_to_one_hot(padded_line, alphabet_size)
    return seqs


if __name__ == '__main__':
    num_chars = 128

    # Learn to predict our own source code.
    text_filename = join(dirname(__file__), 'rnn.py')
    train_inputs = build_dataset(text_filename, sequence_length=30,
                                 alphabet_size=num_chars, max_lines=60)

    init_params = create_rnn_params(input_size=128, output_size=128,
                                    state_size=40, param_scale=0.01)

    def print_training_prediction(weights):
        print("Training text                         Predicted text")
        logprobs = np.asarray(rnn_predict(weights, train_inputs))
        for t in range(logprobs.shape[1]):
            training_text  = one_hot_to_string(train_inputs[:,t,:])
            predicted_text = one_hot_to_string(logprobs[:,t,:])
            print(training_text.replace('\n', ' ') + "|" +
                  predicted_text.replace('\n', ' '))

    def training_loss(params, iter):
        return -rnn_log_likelihood(params, train_inputs, train_inputs)

    def callback(weights, iter, gradient):
        if iter % 10 == 0:
            print("Iteration", iter, "Train loss:", training_loss(weights, 0))
            print_training_prediction(weights)

    # Build gradient of loss function using autograd.
    training_loss_grad = grad(training_loss)

    print("Training RNN...")
    trained_params = adam(training_loss_grad, init_params, step_size=0.1,
                          num_iters=1000, callback=callback)

    print()
    print("Generating text from RNN...")
    num_letters = 30
    for t in range(20):
        text = ""
        for i in range(num_letters):
            seqs = string_to_one_hot(text, num_chars)[:, np.newaxis, :]
            logprobs = rnn_predict(trained_params, seqs)[-1].ravel()
            text += chr(npr.choice(len(logprobs), p=np.exp(logprobs)))
        print(text)
