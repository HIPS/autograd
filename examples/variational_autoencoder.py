# Implements auto-encoding variational Bayes.

from __future__ import absolute_import, division
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm

from autograd import grad
from optimizers import adam
from data import load_mnist, save_images


def diag_gaussian_log_density(x, mu, log_std):
    return np.sum(norm.logpdf(x, mu, np.exp(log_std)), axis=-1)

def unpack_gaussian_params(params):
    # Params of a diagonal Gaussian.
    D = np.shape(params)[-1] / 2
    mean, log_std = params[:, :D], params[:, D:]
    return mean, log_std

def sample_diag_gaussian(mean, log_std, rs):
    return rs.randn(*mean.shape) * np.exp(log_std) + mean

def bernoulli_log_density(targets, logprobs):
    # Targets must be -1 or 1
    label_probabilities = -np.logaddexp(0, -logprobs*targets)
    return np.sum(label_probabilities, axis=-1)   # Sum across pixels.

def relu(x):    return np.maximum(0, x)
def sigmoid(x): return 0.5 * (np.tanh(x) + 1)

def init_net_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """Build a (weights, biases) tuples for each layer."""
    return [(scale * rs.randn(m, n),   # weight matrix
             scale * rs.randn(n))      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

def batch_normalize(activations):
    mbmean = np.mean(activations, axis=0, keepdims=True)
    return (activations - mbmean) / (np.std(activations, axis=0, keepdims=True) + 1)

def neural_net_predict(params, inputs):
    """Params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix."""
    for W, b in params[:-1]:
        outputs = batch_normalize(np.dot(inputs, W) + b)
        inputs = relu(outputs)
    outW, outb = params[-1]
    outputs = np.dot(inputs, outW) + outb
    return outputs

def nn_predict_gaussian(params, inputs):
    return unpack_gaussian_params(neural_net_predict(params, inputs))

def generate_from_noise(gen_params, num_samples, noise_dim, rs):
    noise = rs.randn(num_samples, noise_dim)
    return sigmoid(neural_net_predict(gen_params, noise))

def p_images_given_latents(gen_params, images, latents):
    preds = neural_net_predict(gen_params, latents)
    return bernoulli_log_density(images, preds)

def vae_lower_bound(gen_params, rec_params, data, rs):
    q_means, q_log_stds = nn_predict_gaussian(rec_params, data)
    latents = sample_diag_gaussian(q_means, q_log_stds, rs)
    q_latents = diag_gaussian_log_density(latents, q_means, q_log_stds)
    p_latents = diag_gaussian_log_density(latents, 0, 1)
    likelihood = p_images_given_latents(gen_params, data, latents)
    return np.mean(p_latents + likelihood - q_latents)


if __name__ == '__main__':
    # Model hyper-parameters
    latent_dim = 10
    data_dim = 784
    gen_layer_sizes = [latent_dim, 300, data_dim]
    rec_layer_sizes = [data_dim, 300, latent_dim * 2]

    # Training parameters
    param_scale = 0.01
    batch_size = 200
    num_epochs = 15
    step_size = 0.001

    print("Loading training data...")
    N, train_images, _, test_images, _ = load_mnist()
    on = train_images > 0.5
    train_images = train_images * 0 - 1
    train_images[on] = 1.0

    init_gen_params = init_net_params(param_scale, gen_layer_sizes)
    init_rec_params = init_net_params(param_scale, rec_layer_sizes)
    combined_init_params = (init_gen_params, init_rec_params)

    num_batches = int(np.ceil(len(train_images) / batch_size))
    def batch_indices(iter):
        idx = iter % num_batches
        return slice(idx * batch_size, (idx+1) * batch_size)

    # Define training objective
    seed = npr.RandomState(0)
    def objective(combined_params, iter):
        idx = batch_indices(iter)
        gen_params, rec_params = combined_params
        return -vae_lower_bound(gen_params, rec_params, train_images[idx], seed) / data_dim

    # Get gradients of objective using autograd.
    objective_grad = grad(objective)

    print("     Epoch     |    Objective  |       Fake probability | Real Probability  ")
    def print_perf(combined_params, iter, grad):
        if iter % 10 == 0:
            gen_params, rec_params = combined_params
            bound = np.mean(objective(combined_params, iter))
            print("{:15}|{:20}".format(iter//num_batches, bound))

            fake_data = generate_from_noise(gen_params, 20, latent_dim, seed)
            save_images(fake_data, 'vae_samples.png', vmin=0, vmax=1)
    # The optimizers provided can optimize lists, tuples, or dicts of parameters.
    optimized_params = adam(objective_grad, combined_init_params, step_size=step_size,
                            num_iters=num_epochs * num_batches, callback=print_perf)



