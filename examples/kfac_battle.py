from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from collections import defaultdict

from data import load_mnist
from autograd.optimizers import sgd, adam
from kfac_pre import kfac, log_joint, accuracy, make_table, init_random_params

if __name__ == '__main__':
    # Model parameters
    layer_sizes = [784, 200, 100, 10]
    l2_reg = 0.

    # Training parameters
    param_scale = 0.1
    batch_size = 256
    num_epochs = 50
    step_sizes = [1e-3, 1e-2]
    optimizers = [sgd, adam]
    lambda_values = [1., 1e-1]

    # Load data
    print("Loading training data...")
    N, train_images, train_labels, test_images,  test_labels = load_mnist()

    # Divide data into batches
    num_batches = int(np.ceil(len(train_images) / batch_size))

    def batch_indices(itr):
        idx = itr % num_batches
        return slice(idx * batch_size, (idx+1) * batch_size)

    get_batch = lambda itr: train_images[batch_indices(itr)]

    # Define training objective as a function of iteration index
    def objective(params, itr):
        idx = batch_indices(itr)
        return -log_joint(params, train_images[idx], train_labels[idx], l2_reg)

    # Set up callback to save results

    results = defaultdict(list)

    def make_callback(key):
        print('Starting test for {}'.format(key))
        print_row = make_table(['Epoch', 'Train objective' ,'Train accuracy', 'Test accuracy'])
        def callback(params, i, gradient):
            if i % num_batches == 0:
                train_obj = -log_joint(params, train_images, train_labels, l2_reg)
                train_acc = accuracy(params, train_images, train_labels)
                test_acc  = accuracy(params, test_images, test_labels)
                scores = (i // num_batches, train_obj, train_acc, test_acc)
                print_row(*scores)
                results[key].append(scores)
        return callback

    # Optimizer battle!

    initialize_params = lambda: init_random_params(param_scale, layer_sizes, npr.RandomState(0))

#     for optimizer in optimizers:
#         for step_size in step_sizes:
#             key = '{} {:.0e}'.format(optimizer.func_name, step_size)
#             callback = make_callback(key)
#             init_params = initialize_params()
#             optimized_params = adam(grad(objective), init_params, step_size=step_size,
#                                     num_iters=num_epochs * num_batches, callback=callback)

    eps = 0.05
    sample_period = 1
    reestimate_period = 5
    update_precond_period = 5
    num_samples = batch_size
    step_size = 1e-3
    lmbda = 0.1

    key = '{} {:.0e} {:.0e} {:.2f} {:d} {:d} {:d}'.format(
        'kfac', step_size, lmbda, eps,
        sample_period, reestimate_period, update_precond_period)
    callback = make_callback(key)
    init_params = initialize_params()
    optimized_params = kfac(
        objective, get_batch, layer_sizes, init_params, step_size=step_size,
        num_iters=num_epochs*num_batches, lmbda=lmbda, eps=eps, num_samples=num_samples,
        sample_period=sample_period, reestimate_period=reestimate_period,
        update_precond_period=update_precond_period, callback=callback)
