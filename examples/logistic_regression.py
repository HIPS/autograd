import autograd.numpy as np
from autograd import grad
from autograd.util import quick_grad_check

def sigmoid(x):
def logistic_predictions(weights, inputs):
    # Outputs probability of a label being true according to logistic model.
    return sigmoid(np.dot(inputs, weights))

def training_loss(weights):



# Check the gradients numerically, just to be safe.

