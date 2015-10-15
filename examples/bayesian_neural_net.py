from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm

from black_box_svi import black_box_variational_inference
from optimizers import adam


def make_nn_funs(layer_sizes, L2_reg, noise_variance, nonlinearity=np.tanh):
    """These functions implement a standard multi-layer perceptron,
    vectorized over both training examples and weight samples."""
    shapes = zip(layer_sizes[:-1], layer_sizes[1:])
    num_weights = sum((m+1)*n for m, n in shapes)

    def unpack_layers(weights):
        num_weight_sets = len(weights)
        for m, n in shapes:
            yield weights[:, :m*n]     .reshape((num_weight_sets, m, n)),\
                  weights[:, m*n:m*n+n].reshape((num_weight_sets, 1, n))
            weights = weights[:, (m+1)*n:]

    def predictions(weights, inputs):
        # weights is shape (num_weight_samples x num_weights)
        # inputs  is shape (num_data_samples x num_datapoints x D)
        for W, b in unpack_layers(weights):
            outputs = np.einsum('mnd,mdo->mno', inputs, W) + b
            inputs = nonlinearity(outputs)
        return outputs

    def logprob(weights, inputs, targets):
        log_prior = -L2_reg * np.sum(weights**2, axis=1)
        predicted_outputs = predictions(weights, inputs)
        log_lik = -np.sum((predicted_outputs - targets)**2, axis=1)[:, 0] / noise_variance
        return log_prior + log_lik

    return num_weights, predictions, logprob


def build_toy_dataset(n_data=40, noise_std=0.1):
    D = 1
    rs = npr.RandomState(0)
    inputs  = np.concatenate([np.linspace(0, 2, num=n_data/2), np.linspace(6, 8, num=n_data/2)])
    targets = np.cos(inputs) + rs.randn(n_data) * noise_std
    inputs = (inputs - 5.0) / 4.0
    inputs  = inputs.reshape((1, len(inputs), D))
    targets = targets.reshape((1, len(targets), D))
    return inputs, targets


if __name__ == '__main__':

    # Specify inference problem by its unnormalized log-posterior.
    rbf = lambda x: norm.pdf(x, 0, 1)
    num_weights, predictions, logprob = \
        make_nn_funs(layer_sizes=[1, 20, 1], L2_reg=0.01, noise_variance = 0.01, nonlinearity=rbf)

    inputs, targets = build_toy_dataset()
    log_posterior = lambda weights, t: logprob(weights, inputs, targets)

    # Build variational objective.
    objective, gradient, unpack_params = \
        black_box_variational_inference(log_posterior, num_weights, num_samples=20)

    # Set up figure.
    fig = plt.figure(figsize=(8,8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.ion()
    plt.show(block=False)

    def callback(params, t, g):
        print("Iteration {} lower bound {}".format(t, -objective(params, t)))
        plt.cla()
        ax.plot(inputs.ravel(), targets.ravel(), 'bx')

        n_view_ticks = 200
        viewlims = [-8, 8]
        plot_inputs = np.linspace(*viewlims, num=n_view_ticks).reshape((1, n_view_ticks, 1))
        mean, cov = unpack_params(params)
        rs = npr.RandomState(0)
        sample_weights = rs.randn(10, num_weights) * np.sqrt(cov) + mean
        sample_outs = predictions(sample_weights, plot_inputs)[:, :, 0].T
        plt.plot(plot_inputs.ravel(), sample_outs)
        ax.set_ylim([-2, 3])
        plt.draw()
        plt.pause(1.0/60.0)

    print("Optimizing variational parameters...")
    rs = npr.RandomState(0)
    init_mean    = rs.randn(num_weights)
    init_log_cov = -5 * np.ones(num_weights)
    init_var_params = np.concatenate([init_mean, init_log_cov])
    variational_params = adam(gradient, init_var_params, step_size=0.1, num_iters=1000, callback=callback)
