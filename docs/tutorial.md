# Autograd tutorial

## Motivation

Here's a common situation in machine learning research. You want to test out a
new model for your data. This means coming up with some loss function to capture
how well your model fits the data, and optimizing that loss with respect to the
model parameters. If there are many model parameters (neural nets can have
millions) then you need gradients. You then have two options: derive and code up
them up yourself, or implemented your model using the syntactic and semantic
constraints of a system like Theano. We want to provide a third way: just write
down the loss function using a standard numerical library like numpy, and have
autograd take the gradients for you.

## What is reverse mode differentiation?
* Also known as backpropagation.
* Chain rule evaluated LIFO
* Particularly nice since you don't need to instantiate Jacobians, and
  Jacobian-vector products can often be computed efficiently
* Much faster than forward mode or finite differences
* Much simpler than symbolic differentiation

## How to use autograd
* Grad of a function with loops, etc
* Must have scalar output

## Complete example: logistic regression
* Do a grad check
* For more, see our examples

## What's going on under the hood?
* Node type created
* Operated on by a bunch of primitives
* These are recorded, and the gradients are applied LIFO.

## What can autograd differentiate?
* Keep in mind that it's just a series of function compositions
* The only things acting on the Node and its descendants
  must be primitive, have gradient implemented
* Input can be a vector, tuple, etc

## Supported and unsupported parts of numpy/scipy
* Numpy has a lot of features. Done our best to support most of them but somet
  things remain. For example, indexing we do support, but not assignment
  (assignment is hard because it means you can't rely on data). Similarly,
  binary array methods like ndarray.dot aren't supported (we can't change
  ndarray itself. We experimented with subclassing but this has a host of other
  issues). Assignment to regular arrays is supported but be careful!

#### TL;DR: Do use
* [Most](../autograd/numpy/numpy_grads.py) of numpy's functions
* [Most](../autograd/numpy/numpy_extra.py) numpy.ndarray methods
* [Some](../autograd/scipy/scipy_grads.py) scipy functions
* Indexing and slicing of arrays `x = A[3, :, 2:4]`
* Explicit array creation from lists `A = np.array([x, y])`

#### Don't use
* Assignment to arrays `A[0,0] = x`
* Implicit casting to arrays `A = np.exp([x, y])`
* `A.dot(B)` notation (use `np.dot(A, B)` instead)

## Extend Autograd by defining your own primitives
* Simple example
* What if we have an unknown number of args

## Features and gotchas

### TL;DR

### Planned features
* GPU support
* In-place array operations and assignment to arrays
