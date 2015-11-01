from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm
from autograd import grad

from optimizers import adam


def make_nn_funs(layer_sizes, weight_scale=10.0, noise_scale=0.1, nonlinearity=np.tanh):
    """These functions implement a standard multi-layer perceptron."""
    shapes = zip(layer_sizes[:-1], layer_sizes[1:])
    num_weights = sum((m+1)*n for m, n in shapes)

    def unpack_layers(weights):
        for m, n in shapes:
            cur_layer_weights = weights[:m*n]     .reshape((m, n))
            cur_layer_biases  = weights[m*n:m*n+n].reshape((1, n))
            yield cur_layer_weights, cur_layer_biases
            weights = weights[(m+1)*n:]

    def predictions(weights, inputs):
        for W, b in unpack_layers(weights):
            outputs = np.dot(inputs, W) + b
            inputs = nonlinearity(outputs)
        return outputs

    def logprob(weights, inputs, targets):
        log_prior = np.sum(norm.logpdf(weights, 0, weight_scale))
        preds = predictions(weights, inputs)
        log_lik = np.sum(norm.logpdf(preds, targets, noise_scale))
        return log_prior + log_lik

    return num_weights, predictions, logprob


def build_toy_dataset(n_data=80, noise_std=0.1, D=1):
    rs = npr.RandomState(0)
    inputs  = np.concatenate([np.linspace(0, 3, num=n_data/2),
                              np.linspace(6, 8, num=n_data/2)])
    targets = np.cos(inputs) + rs.randn(n_data) * noise_std
    inputs = (inputs - 4.0) / 2.0
    inputs  = inputs.reshape((len(inputs), D))
    targets = targets.reshape((len(targets), D)) / 2.0
    return inputs, targets


if __name__ == '__main__':

    # Specify inference problem by its unnormalized log-posterior.
    rbf  = lambda x: np.exp(-x**2)
    relu = lambda x: np.maximum(x, 0.0)

    # Implement a 3-hidden layer neural network.
    num_weights, predictions, logprob = \
        make_nn_funs(layer_sizes=[1, 20, 20, 20, 1], nonlinearity=rbf)

    inputs, targets = build_toy_dataset()
    objective = lambda weights, t: -logprob(weights, inputs, targets)

    # Set up figure.
    fig = plt.figure(figsize=(12,8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.show(block=False)

    def callback(params, t, g):
        print("Iteration {} log likelihood {}".format(t, -objective(params, t)))

        # Plot data and functions.
        plt.cla()
        ax.plot(inputs.ravel(), targets.ravel(), 'bx')
        plot_inputs = np.reshape(np.linspace(-7, 7, num=300), (300,1))
        outputs = predictions(params, plot_inputs)
        ax.plot(plot_inputs, outputs)
        ax.set_ylim([-1, 1])
        plt.draw()
        plt.pause(1.0/60.0)

    rs = npr.RandomState(0)
    init_params = 0.1 * rs.randn(num_weights)

    print("Optimizing network parameters...")
    optimized_params = adam(grad(objective), init_params,
                            step_size=0.01, num_iters=1000, callback=callback)
