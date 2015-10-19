from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.multivariate_normal as mvn
import autograd.scipy.stats.norm as norm

from autograd import grad
from optimizers import adam


def black_box_variational_inference(logprob, D, num_samples):
    """Implements http://arxiv.org/abs/1401.0118, and uses the
    local reparameterization trick from http://arxiv.org/abs/1506.02557"""

    def unpack_params(params):
        # Variational dist is a diagonal Gaussian
        # parametrized by its mean and log-variances.
        mean, cov = params[:D], np.exp(params[D:])
        return mean, cov

    rs = npr.RandomState(0)
    def variational_objective(params, t):
        """Provides a stochastic estimate of the variational lower bound."""
        mean, cov = unpack_params(params)
        samples = rs.rand(num_samples, D) * np.sqrt(cov) + mean
        lower_bound = mvn.entropy(mean, np.diag(cov)) + np.mean(logprob(samples, t))
        return -lower_bound

    gradient = grad(variational_objective)

    return variational_objective, gradient, unpack_params



if __name__ == '__main__':

    # Specify an inference problem by its unnormalized log-posterior.
    D = 2
    def log_posterior(x, t):
        """An example 2D intractable distribution:
        a Gaussian evaluated at zero with a Gaussian prior on the log-variance."""
        mu, log_sigma = x[:, 0], x[:, 1]
        prior       = norm.logpdf(log_sigma, 0, 1.35)
        likelihood  = norm.logpdf(mu,        0, np.exp(log_sigma))
        return prior + likelihood

    # Build variational objective.
    objective, gradient, unpack_params = \
        black_box_variational_inference(log_posterior, D, num_samples=2000)

    # Set up plotting code
    def plot_isocontours(ax, func, xlimits=[-2, 2], ylimits=[-4, 2], numticks=101):
        x = np.linspace(*xlimits, num=numticks)
        y = np.linspace(*ylimits, num=numticks)
        X, Y = np.meshgrid(x, y)
        zs = func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T)
        Z = zs.reshape(X.shape)
        plt.contour(X, Y, Z)
        ax.set_yticks([])
        ax.set_xticks([])

    # Set up figure.
    fig = plt.figure(figsize=(8,8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.ion()
    plt.show(block=False)

    def callback(params, t, g):
        print("Iteration {} lower bound {}".format(t, -objective(params, t)))

        plt.cla()
        target_distribution = lambda x : np.exp(log_posterior(x, t))
        plot_isocontours(ax, target_distribution)

        mean, cov = unpack_params(params)
        variational_contour = lambda x: mvn.pdf(x, mean, np.diag(cov))
        plot_isocontours(ax, variational_contour)
        plt.draw()
        plt.pause(1.0/30.0)

    print("Optimizing variational parameters...")
    init_mean    = -1  * np.ones(D)
    init_log_cov = -10 * np.ones(D)
    init_var_params = np.concatenate([init_mean, init_log_cov])
    variational_params = adam(gradient, init_var_params, step_size=0.1, num_iters=2000, callback=callback)
