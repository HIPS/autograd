# Implements black-box variational inference, where the variational
# distribution is a mixture of Gaussians.
#
# This trick was written up by Alex Graves in this note:
# http://arxiv.org/abs/1607.05690

from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm
from autograd.scipy.misc import logsumexp

from autograd import grad
from autograd.optimizers import adam

def diag_gaussian_log_density(x, mu, log_std):
    return np.sum(norm.logpdf(x, mu, np.exp(log_std)), axis=-1)

def unpack_gaussian_params(params):
    # Variational dist is a diagonal Gaussian.
    D = np.shape(params)[0] // 2
    mean, log_std = params[:D], params[D:]
    return mean, log_std

def variational_log_density_gaussian(params, x):
    mean, log_std = unpack_gaussian_params(params)
    return diag_gaussian_log_density(x, mean, log_std)

def sample_diag_gaussian(params, num_samples, rs):
    mean, log_std = unpack_gaussian_params(params)
    D = np.shape(mean)[0]
    return rs.randn(num_samples, D) * np.exp(log_std) + mean

def variational_lower_bound(params, t, logprob, sampler, log_density,
                            num_samples, rs):
    """Provides a stochastic estimate of the variational lower bound,
       for any variational family and model density."""
    samples = sampler(params, num_samples, rs)
    log_qs = log_density(params, samples)
    log_ps = logprob(samples, t)
    log_ps = np.reshape(log_ps, (num_samples, -1))
    log_qs = np.reshape(log_qs, (num_samples, -1))
    return np.mean(log_ps - log_qs)

def init_gaussian_var_params(D, mean_mean=-1, log_std_mean=-5,
                             scale=0.1, rs=npr.RandomState(0)):
    init_mean    = mean_mean * np.ones(D) + rs.randn(D) * scale
    init_log_std = log_std_mean * np.ones(D) + rs.randn(D) * scale
    return np.concatenate([init_mean, init_log_std])

def log_normalize(x):
    return x - logsumexp(x)

def build_mog_bbsvi(logprob, num_samples, k=10, rs=npr.RandomState(0)):
    init_component_var_params = init_gaussian_var_params
    component_log_density = variational_log_density_gaussian
    component_sample = sample_diag_gaussian

    def unpack_mixture_params(mixture_params):
        log_weights = log_normalize(mixture_params[:k])
        var_params = np.reshape(mixture_params[k:], (k, -1))
        return log_weights, var_params

    def init_var_params(D, rs=npr.RandomState(0), **kwargs):
        log_weights = np.ones(k)
        component_weights = [init_component_var_params(D, rs=rs, **kwargs) for i in range(k)]
        return np.concatenate([log_weights] + component_weights)

    def sample(var_mixture_params, num_samples, rs):
        """Sample locations aren't a continuous function of parameters
        due to multinomial sampling."""
        log_weights, var_params = unpack_mixture_params(var_mixture_params)
        samples = np.concatenate([component_sample(params_k, num_samples, rs)[:, np.newaxis, :]
                             for params_k in var_params], axis=1)
        ixs = np.random.choice(k, size=num_samples, p=np.exp(log_weights))
        return np.array([samples[i, ix, :] for i, ix in enumerate(ixs)])

    def mixture_log_density(var_mixture_params, x):
        """Returns a weighted average over component densities."""
        log_weights, var_params = unpack_mixture_params(var_mixture_params)
        component_log_densities = np.vstack([component_log_density(params_k, x)
                                             for params_k in var_params]).T
        return logsumexp(component_log_densities + log_weights, axis=1, keepdims=False)

    def mixture_elbo(var_mixture_params, t):
        # We need to only sample the continuous component parameters,
        # and integrate over the discrete component choice

        def mixture_lower_bound(params):
            """Provides a stochastic estimate of the variational lower bound."""
            samples = component_sample(params, num_samples, rs)
            log_qs = mixture_log_density(var_mixture_params, samples)
            log_ps = logprob(samples, t)
            log_ps = np.reshape(log_ps, (num_samples, -1))
            log_qs = np.reshape(log_qs, (num_samples, -1))
            return np.mean(log_ps - log_qs)

        log_weights, var_params = unpack_mixture_params(var_mixture_params)
        component_elbos = np.stack(
            [mixture_lower_bound(params_k) for params_k in var_params])
        return np.sum(component_elbos*np.exp(log_weights))

    return init_var_params, mixture_elbo, mixture_log_density, sample


if __name__ == '__main__':

    # Specify an inference problem by its unnormalized log-density.
    D = 2
    def log_density(x, t):
        mu, log_sigma = x[:, 0], x[:, 1]
        sigma_density = norm.logpdf(log_sigma, 0, 1.35)
        mu_density = norm.logpdf(mu, -0.5, np.exp(log_sigma))
        sigma_density2 = norm.logpdf(log_sigma, 0.1, 1.35)
        mu_density2 = norm.logpdf(mu, 0.5, np.exp(log_sigma))
        return np.logaddexp(sigma_density + mu_density,
                            sigma_density2 + mu_density2)


    init_var_params, elbo, variational_log_density, variational_sampler = \
        build_mog_bbsvi(log_density, num_samples=40, k=10)

    def objective(params, t):
        return -elbo(params, t)

    # Set up plotting code
    def plot_isocontours(ax, func, xlimits=[-2, 2], ylimits=[-4, 2],
                         numticks=101, cmap=None):
        x = np.linspace(*xlimits, num=numticks)
        y = np.linspace(*ylimits, num=numticks)
        X, Y = np.meshgrid(x, y)
        zs = func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T)
        Z = zs.reshape(X.shape)
        plt.contour(X, Y, Z, cmap=cmap)
        ax.set_yticks([])
        ax.set_xticks([])

    fig = plt.figure(figsize=(8,8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.ion()
    plt.show(block=False)

    num_plotting_samples = 51

    def callback(params, t, g):
        print("Iteration {} lower bound {}".format(t, -objective(params, t)))

        plt.cla()
        target_distribution = lambda x: np.exp(log_density(x, t))
        var_distribution    = lambda x: np.exp(variational_log_density(params, x))
        plot_isocontours(ax, target_distribution)
        plot_isocontours(ax, var_distribution, cmap=plt.cm.bone)
        ax.set_autoscale_on(False)

        rs = npr.RandomState(0)
        samples = variational_sampler(params, num_plotting_samples, rs)
        plt.plot(samples[:, 0], samples[:, 1], 'x')

        plt.draw()
        plt.pause(1.0/30.0)

    print("Optimizing variational parameters...")
    variational_params = adam(grad(objective), init_var_params(D), step_size=0.1,
                              num_iters=2000, callback=callback)
