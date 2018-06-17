"""A multi-layer perceptron for classification of MNIST handwritten digits."""
from __future__ import absolute_import, division, print_function
import numpy as np
import autograd.cupy as cp
import autograd.cupy.random as cpr
from autograd.cupy.autograd_util import flatten, flatten_func
from autograd import grad
from data import load_mnist

### model definition

def init_random_params(scale, layer_sizes):
    """Build a list of (weights, biases) tuples,
       one for each layer in the net."""
    return [(scale * cpr.randn(m, n),   # weight matrix
             scale * cpr.randn(n))      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

def neural_net_predict(params, inputs):
    """Implements a deep neural network for classification.
       params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix.
       returns normalized class log-probabilities."""
    for W, b in params:
        outputs = cp.dot(inputs, W) + b
        inputs = cp.tanh(outputs)
    return outputs - logsumexp(outputs, axis=1, keepdims=True)

def logsumexp(x, axis, keepdims=False):
    amax = cp.max(x, axis=axis, keepdims=keepdims)
    return cp.log(cp.sum(cp.exp(x - amax), axis, keepdims=keepdims)) + amax

### loss functions

def l2_norm(params):
    """Computes l2 norm of params by flattening them into a vector."""
    flattened, _ = flatten(params)
    return cp.dot(flattened, flattened)

def log_posterior(params, inputs, targets, L2_reg):
    log_prior = -L2_reg * l2_norm(params)
    log_lik = cp.sum(neural_net_predict(params, inputs) * targets)
    return log_lik + log_prior

def accuracy(params, inputs, targets):
    target_class    = cp.argmax(targets, axis=1)
    predicted_class = cp.argmax(neural_net_predict(params, inputs), axis=1)
    return cp.mean(predicted_class == target_class)

### optimizer

def rmsprop(grad, init_params, callback=None, num_iters=100,
            step_size=0.1, gamma=0.9, eps=10**-8):
    """Root mean squared prop: See Adagrad paper for details."""
    flattened_grad, unflatten, x = flatten_func(grad, init_params)

    avg_sq_grad = cp.ones(len(x))
    for i in range(num_iters):
        g = flattened_grad(x, i)
        if callback: callback(i, lambda: (unflatten(x), unflatten(g)))
        avg_sq_grad = avg_sq_grad * gamma + g**2 * (1 - gamma)
        x -= step_size * g / (cp.sqrt(avg_sq_grad) + eps)
    return unflatten(x)

### script

if __name__ == '__main__':
    # Model parameters (NOTE bigger layer size than examples/neural_net.py)
    layer_sizes = [784, 1024, 1024, 10]
    L2_reg = 1.0

    # Training parameters
    param_scale = 0.1
    batch_size = 256
    num_epochs = 5
    step_size = 0.001

    print("Loading training data...")
    mnist = load_mnist()
    N = mnist[0]
    train_images, train_labels, test_images, test_labels = map(cp.array, mnist[1:])

    init_params = init_random_params(param_scale, layer_sizes)

    num_batches = int(np.ceil(len(train_images) / batch_size))
    def batch_indices(iter):
        idx = iter % num_batches
        return slice(idx * batch_size, (idx+1) * batch_size)

    # Define training objective
    def objective(params, iter):
        idx = batch_indices(iter)
        return -log_posterior(params, train_images[idx], train_labels[idx], L2_reg)

    # Get gradient of objective using autograd.
    objective_grad = grad(objective)

    print("     Epoch     |    Train accuracy  |       Test accuracy  ")
    def print_perf(iter, thunk):
        if iter % num_batches == 0:
            params, gradient = thunk()
            train_acc = accuracy(params, train_images, train_labels)
            test_acc  = accuracy(params, test_images, test_labels)
            print("{:15}|{:20}|{:20}".format(iter//num_batches, train_acc, test_acc))

    optimized_params = rmsprop(objective_grad, init_params, step_size=step_size,
                               num_iters=num_epochs * num_batches, callback=print_perf)
