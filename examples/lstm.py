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

def activations(weights, *args):
    cat_state = np.concatenate(args + (np.ones((args[0].shape[0],1)),), axis=1)
    return np.dot(cat_state, weights)

def sigmoid(x):
    return 0.5*(np.tanh(x) + 1.0)   # Output ranges from 0 to 1.

def logsumexp(X, axis=1):
    max_X = np.max(X)
    return max_X + np.log(np.sum(np.exp(X - max_X), axis=axis, keepdims=True))

def build_lstm(input_size, state_size, output_size):
    """Build functions to compute the output of an LSTM."""
    parser = WeightsParser()
    parser.add_shape('forget',  (input_size + 2 * state_size + 1, state_size))
    parser.add_shape('change',  (input_size + 2 * state_size + 1, state_size))
    parser.add_shape('ingate',  (input_size + 2 * state_size + 1, state_size))
    parser.add_shape('outgate', (input_size + 2 * state_size + 1, state_size))
    parser.add_shape('predict', (state_size + 1, output_size))

    def update_lstm(input, hidden, cell, forget_weights, change_weights,
                                         ingate_weights, outgate_weights):
        """One iteration of an LSTM layer."""
        change  = np.tanh(activations(change_weights, input, cell, hidden))
        forget  = sigmoid(activations(forget_weights, input, cell, hidden))
        ingate  = sigmoid(activations(ingate_weights, input, cell, hidden))
        cell    = cell * forget + ingate * change
        outgate = sigmoid(activations(outgate_weights, input, cell, hidden))
        hidden = outgate * np.tanh(cell)
        return hidden, cell

    def outputs(weights, inputs):
        """Goes from right to left, updating the state."""
        num_sequences = inputs.shape[1]
        hiddens = np.zeros((num_sequences, state_size))
        cells = np.zeros((num_sequences, state_size))
        output = []
        forget_weights = parser.get(weights, 'forget')
        change_weights = parser.get(weights, 'change')
        ingate_weights = parser.get(weights, 'ingate')
        outgate_weights = parser.get(weights, 'outgate')
        predict_weights = parser.get(weights, 'predict')
        for cur_input in inputs:  # Iterate over time steps.
            hiddens, cells = update_lstm(cur_input, hiddens, cells, forget_weights,
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

    return outputs, loss, frac_err, parser.N

def one_hot_ascii(x, K):
    ascii = np.array([ord(c) for c in x]).T
    return np.array(ascii[:,None] == np.arange(K)[None, :], dtype=int)

def build_dataset(filename, sequence_length, alphabet_size, pad=""):
    with open(filename) as f:
        content = f.readlines()
    content = [line for line in content if len(line) > 2]
    seqs = np.zeros((sequence_length, len(content), alphabet_size))
    for ix, line in enumerate(content):
        padded_line = (pad + line + " " * sequence_length)[:sequence_length]
        seqs[:, ix, :] = one_hot_ascii(padded_line, alphabet_size)
    print "avg spaces ", np.mean(seqs[:,:,ord(' ')])
    return seqs

def make_batches(N_total, N_batch):
    start = 0
    batches = []
    while start < N_total:
        batches.append(slice(start, start + N_batch))
        start += N_batch
    return batches

def demo_lstm():
    npr.seed(1)
    input_size = output_size = 128   # The first 128 ASCII characters are the common ones.
    state_size = 40
    seq_length = 30
    param_scale = 0.01
    train_iters = 100
    minibatch_size = 200

    #train_inputs  = build_dataset('hamlet.txt'      ,  seq_length, input_size, pad = " ")
    #train_targets = build_dataset('hamlet.txt'      ,  seq_length, input_size)
    train_inputs   = build_dataset('lstm.py'      ,  seq_length, input_size, pad = " ")
    train_targets  = build_dataset('lstm.py'      ,  seq_length, input_size)
    #test_inputs   = build_dataset('allswell.txt',  seq_length, input_size, pad = " ")
    #test_targets  = build_dataset('allswell.txt',  seq_length, input_size)

    pred_fun, loss_fun, frac_err, num_weights = build_lstm(input_size, state_size, output_size)

    loss_grad = grad(loss_fun)   # Specifies gradient of loss function using autograd.

    def training_grad_ixs(weights, ix):
        return loss_grad(weights, train_inputs[:, ix, :], train_targets[:, ix, :])
    def training_grad(weights):
        return loss_grad(weights, train_inputs, train_targets)
    def training_loss(weights):
        return loss_fun( weights, train_inputs, train_targets)
    #def training_loss_ix(weights, ix):
    #    return loss_fun( weights, test_inputs[:, ix, :], test_targets[:, ix, :])


    #print "Random error rate:", frac_err(init_weights, test_inputs, test_targets)
    #print "Random gradient:", np.sum(loss_grad(init_weights, train_inputs, train_targets))

    def callback(weights):
        #print "Train loss:", loss_fun(weights, train_inputs, train_targets), \
        #      "Train error:", frac_err(weights, train_inputs, train_targets), \
        #      "Test error:", frac_err(weights, test_inputs,  test_targets)
        print "Train loss:", loss_fun(weights, train_inputs, train_targets), \
              "Train error:", frac_err(weights, train_inputs, train_targets)
              #"Test error:", frac_err(weights, test_inputs[:, ix, :],  test_targets[:, ix, :])

    weights = npr.randn(num_weights) * param_scale
    trained_weights = fmin_cg(training_loss, weights, fprime=training_grad,
                              maxiter=train_iters, callback=callback)


    # Train with sgd
    #N_data = train_inputs.shape[1]
    #batch_idxs = make_batches(N_data, minibatch_size)
    #cur_dir = np.zeros(num_weights)
    #learning_rate = 0.2
    #momentum = np.zeros(num_weights)
    #for epoch in range(train_iters):
    #    callback(weights)
    #    for idxs in batch_idxs:
    #        grad_W = training_grad_ixs(weights, idxs)
    #        cur_dir = momentum * cur_dir + (1.0 - momentum) * grad_W
    #        weights -= learning_rate * cur_dir

    # Generate text
    def logprob_sample(logprobs):
        probs = np.exp(logprobs)
        return npr.choice(len(probs), p=probs)

    num_letters = 60
    for t in xrange(20):
        text = " def"
        for i in xrange(num_letters):
            seqs = np.zeros((len(text), 1, output_size))
            seqs[:,0,:] = one_hot_ascii(text, output_size)
            logprobs = pred_fun(weights, seqs)[-1].ravel()
            text += chr(logprob_sample(logprobs))
        print text

demo_lstm()