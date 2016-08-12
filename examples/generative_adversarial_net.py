# Implements a Generative Adversarial Network, from
# arxiv.org/abs/1406.2661
# but, it always collapses to generating a single image.
# Let me know if you can get it to work! - David Duvenaud

from __future__ import absolute_import, division
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import multigrad
from autograd.util import flatten

from data import load_mnist, save_images


### Define geneerator, discriminator, and objective ###

def relu(x):       return np.maximum(0, x)
def sigmoid(x):    return 0.5 * (np.tanh(x) + 1.0)
def logsigmoid(x): return x - np.logaddexp(0, x)

def init_random_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """Build a list of (weights, biases) tuples,
       one for each layer in the net."""
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

def generate_from_noise(gen_params, num_samples, noise_dim, rs):
    noise = rs.rand(num_samples, noise_dim)
    samples = neural_net_predict(gen_params, noise)
    return sigmoid(samples)

def gan_objective(gen_params, dsc_params, real_data, num_samples, noise_dim, rs):
    fake_data = generate_from_noise(gen_params, num_samples, noise_dim, rs)
    logprobs_fake = logsigmoid(neural_net_predict(dsc_params, fake_data))
    logprobs_real = logsigmoid(neural_net_predict(dsc_params, real_data))
    return np.mean(logprobs_real) - np.mean(logprobs_fake)


### Define minimax version of adam optimizer ###

def adam_minimax(grad_both, init_params_max, init_params_min, callback=None, num_iters=100,
         step_size_max=0.001, step_size_min=0.001, b1=0.9, b2=0.999, eps=10**-8):
    """Adam modified to do minimiax optimization, for instance to help with
    training generative adversarial networks."""

    x_max, unflatten_max = flatten(init_params_max)
    x_min, unflatten_min = flatten(init_params_min)

    m_max = np.zeros(len(x_max))
    v_max = np.zeros(len(x_max))
    m_min = np.zeros(len(x_min))
    v_min = np.zeros(len(x_min))
    for i in range(num_iters):
        g_max_uf, g_min_uf = grad_both(unflatten_max(x_max),
                                       unflatten_min(x_min), i)
        g_max, _ = flatten(g_max_uf)
        g_min, _ = flatten(g_min_uf)

        if callback: callback(unflatten_max(x_max), unflatten_min(x_min), i,
                              unflatten_max(g_max), unflatten_min(g_min))

        m_max = (1 - b1) * g_max      + b1 * m_max  # First  moment estimate.
        v_max = (1 - b2) * (g_max**2) + b2 * v_max  # Second moment estimate.
        mhat_max = m_max / (1 - b1**(i + 1))    # Bias correction.
        vhat_max = v_max / (1 - b2**(i + 1))
        x_max = x_max + step_size_max * mhat_max / (np.sqrt(vhat_max) + eps)

        m_min = (1 - b1) * g_min      + b1 * m_min  # First  moment estimate.
        v_min = (1 - b2) * (g_min**2) + b2 * v_min  # Second moment estimate.
        mhat_min = m_min / (1 - b1**(i + 1))    # Bias correction.
        vhat_min = v_min / (1 - b2**(i + 1))
        x_min = x_min - step_size_min * mhat_min / (np.sqrt(vhat_min) + eps)
    return unflatten_max(x_max), unflatten_min(x_min)


### Setup and run on MNIST ###

if __name__ == '__main__':
    # Model hyper-parameters
    noise_dim = 10
    gen_layer_sizes = [noise_dim, 200, 784]
    dsc_layer_sizes = [784, 200, 1]

    # Training parameters
    param_scale = 0.001
    batch_size = 100
    num_epochs = 50
    step_size_max = 0.01
    step_size_min = 0.01

    print("Loading training data...")
    N, train_images, _, test_images, _ = load_mnist()

    init_gen_params = init_random_params(param_scale, gen_layer_sizes)
    init_dsc_params = init_random_params(param_scale, dsc_layer_sizes)

    num_batches = int(np.ceil(len(train_images) / batch_size))
    def batch_indices(iter):
        idx = iter % num_batches
        return slice(idx * batch_size, (idx+1) * batch_size)

    # Define training objective
    seed = npr.RandomState(0)
    def objective(gen_params, dsc_params, iter):
        idx = batch_indices(iter)
        return gan_objective(gen_params, dsc_params, train_images[idx],
                             batch_size, noise_dim, seed)

    # Get gradients of objective using autograd.
    both_objective_grad = multigrad(objective, argnums=[0,1])

    print("     Epoch     |    Objective  |       Fake probability | Real Probability  ")
    def print_perf(gen_params, dsc_params, iter, gen_gradient, dsc_gradient):
        if iter % 10 == 0:
            ability = np.mean(objective(gen_params, dsc_params, iter))
            fake_data = generate_from_noise(gen_params, 20, noise_dim, seed)
            real_data = train_images[batch_indices(iter)]
            probs_fake = np.mean(sigmoid(neural_net_predict(dsc_params, fake_data)))
            probs_real = np.mean(sigmoid(neural_net_predict(dsc_params, real_data)))
            print("{:15}|{:20}|{:20}|{:20}".format(iter//num_batches, ability, probs_fake, probs_real))
            save_images(fake_data, 'gan_samples.png', vmin=0, vmax=1)

    # The optimizers provided can optimize lists, tuples, or dicts of parameters.
    optimized_params = adam_minimax(both_objective_grad,
                                    init_gen_params, init_dsc_params,
                                    step_size_max=step_size_max, step_size_min=step_size_min,
                                    num_iters=num_epochs * num_batches, callback=print_perf)
