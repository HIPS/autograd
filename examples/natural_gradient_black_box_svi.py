from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.multivariate_normal as mvn
import autograd.scipy.stats.norm as norm

from autograd import grad
from optimizers import adam, sgd

# same BBSVI function!
from black_box_svi import black_box_variational_inference

if __name__ == '__main__':

    # Specify an inference problem by its unnormalized log-density.
    # it's difficult to see the benefit in low dimensions
    # model parameters are a mean and a log_sigma
    np.random.seed(42)
    obs_dim = 20
    Y = np.random.randn(obs_dim, obs_dim).dot(np.random.randn(obs_dim))
    def log_density(x, t):
        mu, log_sigma = x[:, :obs_dim], x[:, obs_dim:]
        sigma_density = np.sum(norm.logpdf(log_sigma, 0, 1.35), axis=1)
        mu_density    = np.sum(norm.logpdf(Y, mu, np.exp(log_sigma)), axis=1)
        return sigma_density + mu_density

    # Build variational objective.
    D = obs_dim * 2    # dimension of our posterior
    objective, gradient, unpack_params = \
        black_box_variational_inference(log_density, D, num_samples=2000)

    # Define the natural gradient
    #   The natural gradient of the ELBO is the gradient of the elbo,
    #   preconditioned by the inverse Fisher Information Matrix.  The Fisher,
    #   in the case of a diagonal gaussian, is a diagonal matrix that is a
    #   simple function of the variance.  Intuitively, statistical distance
    #   created by perturbing the mean of an independent Gaussian is
    #   determined by how wide the distribution is along that dimension ---
    #   the wider the distribution, the less sensitive statistical distances is
    #   to perturbations of the mean; the narrower the distribution, the more
    #   the statistical distance changes when you perturb the mean (imagine
    #   an extremely narrow Gaussian --- basically a spike.  The KL between
    #   this Gaussian and a Gaussian $\epsilon$ away in location can be big ---
    #   moving the Gaussian could significantly reduce overlap in support
    #   which corresponds to a greater statistical distance).
    #
    #   When we want to move in directions of steepest ascent, we multiply by
    #   the inverse fisher --- that way we make quicker progress when the
    #   variance is wide, and we scale down our step size when the variance
    #   is small (which leads to more robust/less chaotic ascent).
    def fisher_diag(lam):
        mu, log_sigma = unpack_params(lam)
        return np.concatenate([np.exp(-2.*log_sigma),
                               np.ones(len(log_sigma))*2])

    # simple! basically free!
    natural_gradient = lambda lam, i: (1./fisher_diag(lam)) * gradient(lam, i)

    # function for keeping track of callback ELBO values (for plotting below)
    def optimize_and_lls(optfun):
        num_iters = 200
        elbos     = []
        def callback(params, t, g):
            elbo_val = -objective(params, t)
            elbos.append(elbo_val)
            if t % 50 == 0:
                print("Iteration {} lower bound {}".format(t, elbo_val))

        init_mean    = -1 * np.ones(D)
        init_log_std = -5 * np.ones(D)
        init_var_params = np.concatenate([init_mean, init_log_std])
        variational_params = optfun(num_iters, init_var_params, callback)
        return np.array(elbos)

    # let's optimize this with a few different step sizes
    elbo_lists = []
    step_sizes = [.1, .25, .5]
    for step_size in step_sizes:
        # optimize with standard gradient + adam
        optfun = lambda n, init, cb: adam(gradient, init, step_size=step_size,
                                                    num_iters=n, callback=cb)
        standard_lls = optimize_and_lls(optfun)

        # optimize with natural gradient + sgd, no momentum
        optnat = lambda n, init, cb: sgd(natural_gradient, init, step_size=step_size,
                                         num_iters=n, callback=cb, mass=.001)
        natural_lls = optimize_and_lls(optnat)
        elbo_lists.append((standard_lls, natural_lls))

    # visually compare the ELBO
    plt.figure(figsize=(12,8))
    colors = ['b', 'k', 'g']
    for col, ss, (stand_lls, nat_lls) in zip(colors, step_sizes, elbo_lists):
        plt.plot(np.arange(len(stand_lls)), stand_lls,
                 '--', label="standard (adam, step-size = %2.2f)"%ss, alpha=.5, c=col)
        plt.plot(np.arange(len(nat_lls)), nat_lls, '-',
                 label="natural (sgd, step-size = %2.2f)"%ss, c=col)

    llrange = natural_lls.max() - natural_lls.min()
    plt.ylim((natural_lls.max() - llrange*.1, natural_lls.max() + 10))
    plt.xlabel("optimization iteration")
    plt.ylabel("ELBO")
    plt.legend(loc='lower right')
    plt.title("%d dimensional posterior"%D)
    plt.show()
