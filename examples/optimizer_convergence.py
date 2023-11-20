from __future__ import absolute_import
from __future__ import print_function
from builtins import range
import autograd.numpy as np
from autograd import grad
from autograd.misc.optimizers import adam
from autograd.test_util import check_grads

def sigmoid(x):
    return 0.5*(np.tanh(x) + 1)

def logistic_predictions(weights, inputs):
    # Outputs probability of a label being true according to logistic model.
    return sigmoid(np.dot(inputs, weights))

def training_loss(weights, t=None):
    # Training loss is the negative log-likelihood of the training labels.
    preds = logistic_predictions(weights, inputs)
    label_probabilities = preds * targets + (1 - preds) * (1 - targets)
    return -np.sum(np.log(label_probabilities))

def callback(params, t, g):
    # The return value of the callback function indicates convergence
    tol = 1e-5
    converged = False
    
    # If the gradients take a very small value, the function has converged
    # Other convergence criteria may be used as well
    if np.sum(np.abs(g)) < tol:
        print("Model converged in iteration {}".format(t))
        converged = True
    
    return converged

# Build a toy dataset.
inputs = np.random.randn(100, 5)
targets = np.random.randint(0, 2, (100, ))

# Build a function that returns gradients of training loss using autograd.
training_gradient_fun = grad(training_loss)

weights = np.zeros((5, ))

# Optimize weights using gradient descent.
print("Initial loss:", training_loss(weights))
weights = adam(training_gradient_fun, weights, step_size=0.1, num_iters=500, callback=callback)
print("Trained loss:", training_loss(weights))
