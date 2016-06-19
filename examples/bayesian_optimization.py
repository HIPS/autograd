"""This Bayesian optimization demo using gradient-based optimization
   to find the next query point."""

from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import value_and_grad
from scipy.optimize import minimize
from gaussian_process import make_gp_funs, rbf_covariance
from autograd.scipy.stats import norm

def probability_of_improvement(mean, std, max_so_far):
    return norm.cdf(max_so_far, mean, std)

def expected_new_max(mean, std, max_so_far):
    return max_so_far - \
           (mean - max_so_far) * norm.cdf(mean, max_so_far, std) \
                         + std * norm.pdf(mean, max_so_far, std)

def init_covariance_params(num_params):
    return np.zeros(num_params)

def defaultmax(x, default=-np.inf):
    if x.size == 0:
        return default
    return np.max(x)

def bayesian_optimize(func, domain_min, domain_max, num_iters=20, callback=None):

    D = len(domain_min)

    num_params, predict, log_marginal_likelihood = \
        make_gp_funs(rbf_covariance, num_cov_params=D + 1)

    model_params = init_covariance_params(num_params)

    def optimize_gp_params(init_params, X, y):
        log_hyperprior = lambda params: np.sum(norm.logpdf(params, 0., 100.))
        objective = lambda params: -log_marginal_likelihood(params, X, y) -log_hyperprior(params)
        return minimize(value_and_grad(objective), init_params, jac=True, method='CG').x

    def choose_next_point(domain_min, domain_max, acquisition_function, num_tries=15, rs=npr.RandomState(0)):
        """Uses gradient-based optimization to find next query point."""
        init_points = rs.rand(num_tries, D) * (domain_max - domain_min) + domain_min

        grad_obj = value_and_grad(lambda x: -acquisition_function(x))
        def optimize_point(init_point):
            print('.', end='')
            result = minimize(grad_obj, x0=init_point, jac=True, method='L-BFGS-B',
                              options={'maxiter': 10}, bounds=list(zip(domain_min, domain_max)))
            return result.x, acquisition_function(result.x)
        optimzed_points, optimized_values = list(zip(*list(map(optimize_point, init_points))))
        print()
        best_ix = np.argmax(optimized_values)
        return np.atleast_2d(optimzed_points[best_ix])


    # Start by evaluating once in the middle of the domain.
    X = np.zeros((0, D))
    y = np.zeros((0))
    X = np.concatenate((X, np.reshape((domain_max - domain_min) / 2.0, (D, 1))))
    y = np.concatenate((y, np.reshape(np.array(func(X)), (1,))))

    for i in range(num_iters):
        if i > 1:
            print("Optimizing model parameters...")
            model_params = optimize_gp_params(model_params, X, y)

        print("Choosing where to look next", end='')
        def predict_func(xstar):
            mean, cov = predict(model_params, X, y, xstar)
            return mean, np.sqrt(np.diag(cov))

        def acquisition_function(xstar):
            xstar = np.atleast_2d(xstar)  # To work around a bug in scipy.minimize
            mean, std = predict_func(xstar)
            return expected_new_max(mean, std, defaultmax(y))
        next_point = choose_next_point(domain_min, domain_max, acquisition_function)

        print("Evaluating expensive function...")
        new_value = func(next_point)

        X = np.concatenate((X, next_point))
        y = np.concatenate((y, np.reshape(np.array(new_value), (1,))))

        if callback:
            callback(X, y, predict_func, acquisition_function, next_point, new_value)

    best_ix = np.argmax(y)
    return X[best_ix, :], y[best_ix]


if __name__ == '__main__':

    def example_function(x):
        return np.sum(x * np.sin(10.0*x) + x) - 1
    domain_min = np.array([0.0])
    domain_max = np.array([1.1])


    # Set up figure.
    fig = plt.figure(figsize=(12,8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.show(block=False)

    def callback(X, y, predict_func, acquisition_function, next_point, new_value):
        plt.cla()

        # Show posterior marginals.
        plot_xs = np.reshape(np.linspace(domain_min, domain_max, 300), (300,1))
        pred_mean, pred_std = predict_func(plot_xs)
        ax.plot(plot_xs, pred_mean, 'b')
        ax.fill(np.concatenate([plot_xs, plot_xs[::-1]]),
                np.concatenate([pred_mean - 1.96 * pred_std,
                               (pred_mean + 1.96 * pred_std)[::-1]]),
                alpha=.15, fc='Blue', ec='None')

        ax.plot(X, y, 'kx')
        ax.plot(next_point, new_value, 'ro')

        alphas = acquisition_function(plot_xs)
        ax.plot(plot_xs, alphas, 'r')
        ax.set_ylim([-1.5, 1.5])
        ax.set_xticks([])
        ax.set_yticks([])
        plt.draw()
        plt.pause(1)

    best_x, best_y = bayesian_optimize(example_function, domain_min, domain_max, callback=callback)

