from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import value_and_grad
from scipy.optimize import minimize

from gaussian_process import make_gp_funs, rbf_covariance

def build_step_function_dataset(D=1, n_data=40, noise_std=0.1):
    rs = npr.RandomState(0)
    inputs  = np.linspace(-2, 2, num=n_data)
    targets = np.sign(inputs) + rs.randn(n_data) * noise_std
    inputs  = inputs.reshape((len(inputs), D))
    return inputs, targets


def build_deep_gp(input_dimension, hidden_dimension, covariance_function):

    # GP going from input to hidden
    num_params_layer1, predict_layer1, log_marginal_likelihood_layer1 = \
        make_gp_funs(covariance_function, num_cov_params=input_dimension + 1)

    # GP going from hidden to output
    num_params_layer2, predict_layer2, log_marginal_likelihood_layer2 = \
        make_gp_funs(covariance_function, num_cov_params=hidden_dimension + 1)

    num_hidden_params = hidden_dimension * n_data
    total_num_params = num_params_layer1 + num_params_layer2 + num_hidden_params

    def unpack_all_params(all_params):
        layer1_params = all_params[:num_params_layer1]
        layer2_params = all_params[num_params_layer1:num_params_layer1+num_params_layer2]
        hiddens = all_params[num_params_layer1 + num_params_layer2:]
        return layer1_params, layer2_params, hiddens

    def combined_predict_fun(all_params, X, y, xs):
        layer1_params, layer2_params, hiddens = unpack_all_params(all_params)
        h_star_mean, h_star_cov = predict_layer1(layer1_params, X, hiddens, xs)
        y_star_mean, y_star_cov = predict_layer2(layer2_params, np.atleast_2d(hiddens).T, y, np.atleast_2d(h_star_mean).T)
        return y_star_mean, y_star_cov

    def log_marginal_likelihood(all_params):
        layer1_params, layer2_params, h = unpack_all_params(all_params)
        return log_marginal_likelihood_layer1(layer1_params, X, h) + \
               log_marginal_likelihood_layer2(layer2_params, np.atleast_2d(h).T, y)

    predict_layer_funcs = [predict_layer1, predict_layer2]

    return total_num_params, log_marginal_likelihood, combined_predict_fun, unpack_all_params, \
           predict_layer_funcs


if __name__ == '__main__':

    n_data = 20
    input_dimension = 1
    hidden_dimension = 1
    X, y = build_step_function_dataset(D=input_dimension, n_data=n_data)

    total_num_params, log_marginal_likelihood, combined_predict_fun, unpack_all_params, predict_layer_funcs = \
        build_deep_gp(input_dimension, hidden_dimension, rbf_covariance)

    # Set up figure.
    fig = plt.figure(figsize=(12,8), facecolor='white')
    ax_end_to_end = fig.add_subplot(311, frameon=False)
    ax_x_to_h = fig.add_subplot(312, frameon=False)
    ax_h_to_y = fig.add_subplot(313, frameon=False)
    plt.show(block=False)

    def plot_gp(ax, X, y, pred_mean, pred_cov, plot_xs):
        ax.cla()
        marg_std = np.sqrt(np.diag(pred_cov))
        ax.plot(plot_xs, pred_mean, 'b')
        ax.fill(np.concatenate([plot_xs, plot_xs[::-1]]),
                np.concatenate([pred_mean - 1.96 * marg_std,
                               (pred_mean + 1.96 * marg_std)[::-1]]),
                alpha=.15, fc='Blue', ec='None')

        # Show samples from posterior.
        rs = npr.RandomState(0)
        sampled_funcs = rs.multivariate_normal(pred_mean, pred_cov, size=10)
        ax.plot(plot_xs, sampled_funcs.T)
        ax.plot(X, y, 'kx')
        ax.set_ylim([-1.5, 1.5])
        ax.set_xticks([])
        ax.set_yticks([])

    def callback(params):
        print("Log marginal likelihood {}".format(log_marginal_likelihood(params)))

        # Show posterior marginals.
        plot_xs = np.reshape(np.linspace(-5, 5, 300), (300,1))
        pred_mean, pred_cov = combined_predict_fun(params, X, y, plot_xs)
        plot_gp(ax_end_to_end, X, y, pred_mean, pred_cov, plot_xs)
        ax_end_to_end.set_title("X to y")

        layer1_params, layer2_params, hiddens = unpack_all_params(params)
        h_star_mean, h_star_cov = predict_layer_funcs[0](layer1_params, X, hiddens, plot_xs)
        y_star_mean, y_star_cov = predict_layer_funcs[0](layer2_params, np.atleast_2d(hiddens).T, y, plot_xs)

        plot_gp(ax_x_to_h, X, hiddens,                  h_star_mean, h_star_cov, plot_xs)
        ax_x_to_h.set_title("X to hiddens")

        plot_gp(ax_h_to_y, np.atleast_2d(hiddens).T, y, y_star_mean, y_star_cov, plot_xs)
        ax_h_to_y.set_title("hiddens to y")

        plt.draw()
        plt.pause(1.0/60.0)

    # Initialize covariance parameters and hiddens.
    rs = npr.RandomState(0)
    init_params = 0.1 * rs.randn(total_num_params)

    print("Optimizing covariance parameters...")
    objective = lambda params: -log_marginal_likelihood(params)
    cov_params = minimize(value_and_grad(objective), init_params, jac=True,
                          method='CG', callback=callback)
    plt.pause(10.0)
