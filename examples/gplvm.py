# Implements a Gaussian process latent-variable model.
# The (high-dimensional) data, Y is explained by some low-dimensional latent
# data X, warped by a function drawn from a GP prior (f).  So Y = f(X), but
# we don't know X or f.
#
# In this example, we optimize X and the hyperparameters of the GP, but
# we integrate over all possible functions f.
#
# Normally the observed data would be high-dimensional.
#
# David Duvenaud (duvenaud@gmail.com)


from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import value_and_grad
from scipy.optimize import minimize
from autograd.scipy.stats import norm

from gaussian_process import make_gp_funs, rbf_covariance
from data import make_pinwheel

if __name__ == '__main__':

    data_dimension = 2   # Normally the data dimension would be much higher.
    latent_dimension = 2

    # Build model and objective function.
    params_per_gp, predict, log_marginal_likelihood = \
        make_gp_funs(rbf_covariance, num_cov_params=latent_dimension + 1)
    total_gp_params = data_dimension * params_per_gp

    data = make_pinwheel(radial_std=0.3, tangential_std=0.05, num_classes=3,
                         num_per_class=30, rate=0.4)
    datalen = data.shape[0]

    num_latent_params = datalen * latent_dimension

    def unpack_params(params):
        gp_params = np.reshape(params[:total_gp_params], (data_dimension, params_per_gp))
        latents   = np.reshape(params[total_gp_params:], (datalen, latent_dimension))
        return gp_params, latents

    def objective(params):
        gp_params, latents = unpack_params(params)
        gp_likelihood = sum([log_marginal_likelihood(gp_params[i], latents, data[:, i])
                             for i in range(data_dimension)])
        latent_prior_likelihood = np.sum(norm.logpdf(latents))
        return -gp_likelihood - latent_prior_likelihood

    # Set up figure.
    fig = plt.figure(figsize=(12,8), facecolor='white')
    latent_ax = fig.add_subplot(121, frameon=False)
    data_ax = fig.add_subplot(122, frameon=False)
    plt.show(block=False)

    def callback(params):
        print("Log likelihood {}".format(-objective(params)))
        gp_params, latents = unpack_params(params)

        data_ax.cla()
        data_ax.plot(data[:, 0], data[:, 1], 'bx')
        data_ax.set_xticks([])
        data_ax.set_yticks([])
        data_ax.set_title('Observed Data')

        latent_ax.cla()
        latent_ax.plot(latents[:,0], latents[:,1], 'kx')
        latent_ax.set_xticks([])
        latent_ax.set_yticks([])
        latent_ax.set_xlim([-2, 2])
        latent_ax.set_ylim([-2, 2])
        latent_ax.set_title('Latent coordinates')

        plt.draw()
        plt.pause(1.0/60.0)

    # Initialize covariance parameters
    rs = npr.RandomState(1)
    init_params = rs.randn(total_gp_params + num_latent_params) * 0.1

    print("Optimizing covariance parameters and latent variable locations...")
    minimize(value_and_grad(objective), init_params, jac=True, method='CG', callback=callback)
