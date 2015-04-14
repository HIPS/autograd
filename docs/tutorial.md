# Autograd tutorial


## Motivation

* Machine learning is often about defining a loss and optimizing it. We want to
  make it so that you just write down the forward computation and optimization
  comes for free
* What is reverse mode differentiation? (a.k.a. backpropagation)
* Is this just symbolic differentiation?
* Why not just use numerical derivatives?

## Using Autograd

### Basic usage
* Grad of a simple function
* Grad of a function with loops, etc
* For more, see our examples
* Must have scalar output
* Do a grad check

### What's going on under the hood?
* Node type created
* Operated on by a bunch of primitives
* These are recorded, and the gradients are applied LIFO.

### What can be differentiated?
* The only things acting on the Node and its descendants
  must be primitive, have gradient implemented
* Input can be a vector, tuple, etc

## Extend Autograd by defining your own primitives
* Simple example
* What if we have an unknown number of args

## Features and gotchas

### TL;DR
#### Do use
* [Most](../autograd/numpy/numpy_grads.py) of numpy's functions
* [Most](../autograd/numpy/numpy_extra.py) numpy.ndarray methods
* [Some](../autograd/scipy/scipy_grads.py) scipy functions
* Indexing and slicing of arrays `x = A[3, :, 2:4]`
* Explicit array creation from lists `A = np.array([x, y])`
#### Don't use
* Assignment to arrays `A[0,0] = x`
* Implicit casting to arrays `A = np.exp([x, y])`
* `A.dot(B)` notation (use `np.dot(A, B)` instead)

### Planned features
* GPU support
* In-place array operations and assignment to arrays
