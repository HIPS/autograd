from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.t as t
from autograd import value_and_grad

from scipy.optimize import minimize

def make_ica_funs(observed_dimension, latent_dimension):
    """These functions implement independent component analysis.

    The model is:
    latents are drawn i.i.d. for each data point from a product of student-ts.
    weights are the same across all datapoints.
    each data = latents * weghts + noise."""

    def sample(weights, n_samples, noise_std, rs):
        latents = rs.randn(latent_dimension, n_samples)
        latents = np.array(sorted(latents.T, key=lambda a_entry: a_entry[0])).T
        noise = rs.randn(n_samples, observed_dimension) * noise_std
        observed = predict(weights, latents) + noise
        return latents, observed

    def predict(weights, latents):
        return np.dot(weights, latents).T

    def logprob(weights, latents, noise_std, observed):
        preds = predict(weights, latents)
        log_lik = np.sum(t.logpdf(preds, 2.4, observed, noise_std))
        return log_lik

    num_weights = observed_dimension * latent_dimension

    def unpack_weights(weights):
        return np.reshape(weights, (observed_dimension, latent_dimension))

    return num_weights, sample, logprob, unpack_weights

def color_scatter(ax, xs, ys):
    colors = cm.rainbow(np.linspace(0, 1, len(ys)))
    for x, y, c in zip(xs, ys, colors):
        ax.scatter(x, y, color=c)


if __name__ == '__main__':

    observed_dimension = 100
    latent_dimension = 2
    true_noise_var = 1.0
    n_samples = 200

    num_weights, sample, logprob, unpack_weights = \
        make_ica_funs(observed_dimension, latent_dimension)

    num_latent_params = latent_dimension * n_samples
    total_num_params = num_weights + num_latent_params + 1
    def unpack_params(params):
        weights = unpack_weights(params[:num_weights])
        latents = np.reshape(params[num_weights:num_weights+num_latent_params], (latent_dimension, n_samples))
        noise_std = np.exp(params[-1])
        return weights, latents, noise_std

    rs = npr.RandomState(0)
    true_weights = np.zeros((observed_dimension, latent_dimension))
    for i in range(latent_dimension):
        true_weights[:,i] = np.sin(np.linspace(0,4 + i*3.2, observed_dimension))

    true_latents, data = sample(true_weights, n_samples, true_noise_var, rs)

    # Set up figure.
    fig2 = plt.figure(figsize=(6, 6), facecolor='white')
    ax_data = fig2.add_subplot(111, frameon=False)
    ax_data.matshow(data)

    fig1 = plt.figure(figsize=(12,16), facecolor='white')
    ax_true_latents = fig1.add_subplot(411, frameon=False)
    ax_est_latents  = fig1.add_subplot(412, frameon=False)
    ax_true_weights = fig1.add_subplot(413, frameon=False)
    ax_est_weights  = fig1.add_subplot(414, frameon=False)

    plt.show(block=False)
    ax_true_weights.scatter(true_weights[:, 0], true_weights[:, 1])
    ax_true_weights.set_title("True weights")
    color_scatter(ax_true_latents, true_latents[0, :], true_latents[1, :])
    ax_true_latents.set_title("True latents")
    ax_true_latents.set_xticks([])
    ax_true_weights.set_xticks([])
    ax_true_latents.set_yticks([])
    ax_true_weights.set_yticks([])

    def objective(params):
        weight_matrix, latents, noise_std = unpack_params(params)
        return -logprob(weight_matrix, latents, noise_std, data)/n_samples

    def callback(params):
        weights, latents, noise_std = unpack_params(params)
        print("Log likelihood {}, noise_std {}".format(-objective(params), noise_std))
        ax_est_weights.cla()
        ax_est_weights.scatter(weights[:, 0], weights[:, 1])
        ax_est_weights.set_title("Estimated weights")
        ax_est_latents.cla()
        color_scatter(ax_est_latents, latents[0, :], latents[1, :])
        ax_est_latents.set_title("Estimated latents")
        ax_est_weights.set_yticks([])
        ax_est_latents.set_yticks([])
        ax_est_weights.set_xticks([])
        ax_est_latents.set_xticks([])
        plt.draw()
        plt.pause(1.0/60.0)

    # Initialize and optimize model.
    rs = npr.RandomState(0)
    init_params = rs.randn(total_num_params)
    minimize(value_and_grad(objective), init_params, jac=True, method='CG', callback=callback)
    plt.pause(20)