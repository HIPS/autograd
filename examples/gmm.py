"""Implements a Gaussian mixture model, in which parameters are fit using
   gradient descent.  This example runs on 2-dimensional data, but the model
   works on arbitrarily-high dimension."""

from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad, hessian_vector_product
from scipy.optimize import minimize
from autograd.scipy.misc import logsumexp
import autograd.scipy.stats.multivariate_normal as mvn
from data import make_pinwheel
from autograd.util import flatten_func


def init_gmm_params(num_components, D, scale, rs=npr.RandomState(0)):
    return {'log proportions': rs.randn(num_components) * scale,
            'means':           rs.randn(num_components, D) * scale,
            'lower triangles': np.zeros((num_components, D, D)) + np.eye(D)}

def log_normalize(x):
    return x - logsumexp(x)

def unpack_gmm_params(params):
    normalized_log_proportions = log_normalize(params['log proportions'])
    return normalized_log_proportions, params['means'], params['lower triangles']

def gmm_log_likelihood(params, data):
    cluster_lls = []
    for log_proportion, mean, cov_sqrt in zip(*unpack_gmm_params(params)):
        cov = np.dot(cov_sqrt.T, cov_sqrt)
        cluster_lls.append(log_proportion + mvn.logpdf(data, mean, cov))
    return np.sum(logsumexp(np.vstack(cluster_lls), axis=0))

def plot_ellipse(ax, mean, cov_sqrt, alpha, num_points=100):
    angles = np.linspace(0, 2*np.pi, num_points)
    circle_pts = np.vstack([np.cos(angles), np.sin(angles)]).T * 2.0
    cur_pts = mean + np.dot(circle_pts, cov_sqrt)
    ax.plot(cur_pts[:, 0], cur_pts[:, 1], '-', alpha=alpha)

def plot_gaussian_mixture(params, ax):
    for log_proportion, mean, cov_sqrt in zip(*unpack_gmm_params(params)):
        alpha = np.minimum(1.0, np.exp(log_proportion) * 10)
        plot_ellipse(ax, mean, cov_sqrt, alpha)

if __name__ == '__main__':

    init_params = init_gmm_params(num_components=10, D=2, scale=0.1)

    data = make_pinwheel(radial_std=0.3, tangential_std=0.05, num_classes=3,
                         num_per_class=100, rate=0.4)

    def objective(params):
        return -gmm_log_likelihood(params, data)

    flattened_obj, unflatten, flattened_init_params =\
        flatten_func(objective, init_params)

    fig = plt.figure(figsize=(12,8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.show(block=False)

    def callback(flattened_params):
        params = unflatten(flattened_params)
        print("Log likelihood {}".format(-objective(params)))
        ax.cla()
        ax.plot(data[:, 0], data[:, 1], 'k.')
        ax.set_xticks([])
        ax.set_yticks([])
        plot_gaussian_mixture(params, ax)
        plt.draw()
        plt.pause(1.0/60.0)

    minimize(flattened_obj, flattened_init_params,
             jac=grad(flattened_obj),
             hessp=hessian_vector_product(flattened_obj),
             method='Newton-CG', callback=callback)

