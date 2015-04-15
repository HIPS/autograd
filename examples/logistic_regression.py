import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.util import quick_grad_check
from scipy.optimize import minimize

def sigmoid(x):
    return 0.5*(np.tanh(x) + 1.0)   # Output ranges from 0 to 1.

# Specify logistic regression model.
def predictions(weights, inputs):
    return sigmoid(np.dot(inputs, weights))   # Outputs probabilities.

# Specify loss function (negative log-likelihood).
def loss(weights, inputs, targets):
    preds = predictions(weights, inputs)
    log_lik = np.sum(np.log(preds) * targets) \
            + np.sum(np.log(1 - preds) * (1 - targets))
    return -log_lik
    
# Generate data.
num_weights = 10
num_data = 100
true_weights = npr.randn(num_weights)
train_inputs = npr.randn(num_data, num_weights)
test_inputs  = npr.randn(num_data, num_weights)
train_targets = npr.rand(num_data) > predictions(true_weights, train_inputs)
test_targets  = npr.rand(num_data) > predictions(true_weights, test_inputs)

def training_loss(weights):
    return loss(weights, train_inputs, train_targets)

# Build a function that returns (training loss, gradients) using autograd.
loss_with_grad = grad(training_loss, return_function_value=True)

# Initialize weights.
init_weights = npr.randn(num_weights)

# Check the gradients numerically, just to be safe.
quick_grad_check(training_loss, init_weights)

result = minimize(loss_with_grad, x0=init_weights, jac=True, method='CG')
trained_weights = result.x
print "Training loss:", loss(trained_weights, train_inputs, train_targets)
print "Testing  loss:", loss(trained_weights, test_inputs, test_targets)
