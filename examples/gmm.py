# Implements a Gaussian mixture model, in which parameters are fit using
# gradient descent.  This example runs on 2-dimensional data, but the model
# works on arbitrarily-high dimension.


from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import value_and_grad
from scipy.optimize import minimize
from autograd.scipy.misc import logsumexp
import autograd.scipy.stats.multivariate_normal as mvn


class WeightsParser(object):
    """A helper class to index into a parameter vector."""
    def __init__(self):
        self.idxs_and_shapes = {}
        self.num_weights = 0

    def add_shape(self, name, shape):
        start = self.num_weights
        self.num_weights += np.prod(shape)
        self.idxs_and_shapes[name] = (slice(start, self.num_weights), shape)

    def get(self, vect, name):
        idxs, shape = self.idxs_and_shapes[name]
        return np.reshape(vect[idxs], shape)


def make_gmm_funcs(num_components, D):

    parser = WeightsParser()
    parser.add_shape('log proportions',  num_components)
    parser.add_shape('means',           (num_components, D))
    parser.add_shape('lower triangles', (num_components, D, D))
    parser.add_shape('log diagonals',   (num_components, D))

    def unpack_params(params):
        """Unpacks parameter vector into the proportions, means and covariances
        of each mixture component.  The covariance matrices are parametrized by
        their Cholesky decompositions."""
        log_proportions    = parser.get(params, 'log proportions')
        normalized_log_proportions = log_proportions - logsumexp(log_proportions)
        means              = parser.get(params, 'means')
        lower_tris = np.tril(parser.get(params, 'lower triangles'), k=-1)
        diag_chols = np.exp( parser.get(params, 'log diagonals'))
        chols = []
        for lower_tri, diag in zip(lower_tris, diag_chols):
            chols.append(np.expand_dims(lower_tri + np.diag(diag), 0))
        chols = np.concatenate(chols, axis=0)
        return normalized_log_proportions, means, chols

    def log_marginal_likelihood(params, data):
        cluster_lls = []
        for log_proportion, mean, chol in zip(*unpack_params(params)):
            cov = np.dot(chol.T, chol) + 0.000001 * np.eye(D)
            cluster_log_likelihood = log_proportion + mvn.logpdf(data, mean, cov)
            cluster_lls.append(np.expand_dims(cluster_log_likelihood, axis=0))
        cluster_lls = np.concatenate(cluster_lls, axis=0)
        return np.sum(logsumexp(cluster_lls, axis=0))

    return parser.num_weights, log_marginal_likelihood, unpack_params


def make_pinwheel_data(num_spokes=5, points_per_spoke=40, rate=1.0, noise_std=0.005):
    """Make synthetic data in the shape of a pinwheel."""
    spoke_angles = np.linspace(0, 2 * np.pi, num_spokes + 1)[:-1]
    rs = npr.RandomState(0)
    x = np.linspace(0.1, 1, points_per_spoke)
    xs = np.concatenate([x * np.cos(angle + x * rate) + noise_std * rs.randn(len(x))
                         for angle in spoke_angles])
    ys = np.concatenate([x * np.sin(angle + x * rate) + noise_std * rs.randn(len(x))
                         for angle in spoke_angles])
    return np.concatenate([np.expand_dims(xs, 1), np.expand_dims(ys,1)], axis=1)


if __name__ == '__main__':

    num_gmm_params, log_marginal_likelihood, unpack_params = \
        make_gmm_funcs(num_components=15, D=2)

    data = make_pinwheel_data()
    def objective(params):
        return -log_marginal_likelihood(params, data)

    def plot_gmm(params, ax, num_points=100):
        angles = np.expand_dims(np.linspace(0, 2*np.pi, num_points), 1)
        xs, ys = np.cos(angles), np.sin(angles)
        circle_pts = np.concatenate([xs, ys], axis=1) * 2.0
        for log_proportion, mean, chol in zip(*unpack_params(params)):
            cur_pts = mean + np.dot(circle_pts, chol)
            ax.plot(cur_pts[:, 0], cur_pts[:, 1], '-')

    fig = plt.figure(figsize=(12,8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.show(block=False)

    def callback(params):
        print("Log likelihood {}".format(-objective(params)))
        ax.cla()
        ax.plot(data[:, 0], data[:, 1], 'bx')
        ax.set_xticks([])
        ax.set_yticks([])
        plot_gmm(params, ax)
        plt.draw()
        plt.pause(1.0/60.0)

    # Initialize and optimize model.
    rs = npr.RandomState(0)
    init_params = rs.randn(num_gmm_params) * 0.1
    minimize(value_and_grad(objective), init_params, jac=True, method='CG', callback=callback)
