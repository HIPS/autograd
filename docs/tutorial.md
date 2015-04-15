# Autograd tutorial

## Motivation

Imagine you want to test out a new machine learning model for your data. This
usually means coming up with some loss function to capture how well your model
fits the data and optimizing that loss with respect to the model parameters. If
there are many model parameters (neural nets can have millions) then you need
gradients. You then have two options: derive and code them up yourself, or
implement your model using the syntactic and semantic constraints of a system
like [Theano](http://deeplearning.net/software/theano/).

We want to provide a third way: just write down the loss function using a
standard numerical library like numpy, and autograd will give you its gradient.

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
* Vector inputs is where the action is
* Do a grad check
* For more, see our [examples directory](../examples/), which includes:
* [a simple neural net](../examples/neural_net.py)
* [a convolutional neural net](../examples/convnet.py)
* [a rcurrent neural net](../examples/rnn.py)
* [a long short-term memory (LSTM)](../examples/lstm.py)
* [backpropagating through a fluid simulation](../examples/fluidsim/fluidsim.py) 

## What's going on under the hood?
* Node type created
* Operated on by a bunch of primitives
* These are recorded, and the gradients are applied LIFO.

## What can autograd differentiate?
Autograd's `grad` method takes in a function, and returns another function that computes its derivative.
* Keep in mind that it's just a series of function compositions
* The only things acting on the Node and its descendants
must be primitive, have gradient implemented
* Input can be a vector, tuple, etc

## Supported and unsupported parts of numpy/scipy

Numpy has [a lot of features](http://docs.scipy.org/doc/numpy/reference/). We've done our best to support most of them.  We've implemented gradients for:
* most of the [mathematical operations](http://docs.scipy.org/doc/numpy/reference/routines.math.html)
* most of the [array manipulation routines](http://docs.scipy.org/doc/numpy/reference/routines.array-manipulation.html)
* some [linear algebra](../autograd/numpy/linalg.py) functions
* most of the [fast fourier transform](http://docs.scipy.org/doc/numpy/reference/routines.fft.html) routines
* full support for complex numbers
* [N-dimensional convolutions](../autograd/scipy/signal.py)
* Some scipy routines, including [`scipy.stats.norm`](../autograd/scipy/stats/norm.py)


However, some things remain to be implemented. For example, we support indexing: `x = A[i, j, :]`, but not assignment: `A[i,j] = x`.
Assignment is hard to support because it means you can't rely on data.

Similarly, one particular way of calling binary array methods like ndarray.dot isn't supported, because we can't change ndarray itself. We experimented with subclassing but this has a host of other issues). Assignment to regular arrays is supported but be careful!

#### TL;DR: Do use
* [Most](../autograd/numpy/numpy_grads.py) of numpy's functions
* [Most](../autograd/numpy/numpy_extra.py) numpy.ndarray methods
* [Some](../autograd/scipy/scipy_grads.py) scipy functions
* Indexing and slicing of arrays `x = A[3, :, 2:4]`
* Explicit array creation from lists `A = np.array([x, y])`

#### Don't use
* Assignment to arrays `A[0,0] = x`
* Implicit casting to arrays `A = np.sum([x, y])`
* `A.dot(B)` notation (use `np.dot(A, B)` instead)

## Extend Autograd by defining your own primitives
What if autograd doesn't support a function I need to take the gradient of?
This can happen if your code depends on external library calls or C code.
It can also be a good idea to provide the gradient of a pure Python function for speed or numerical stability.

For example, let's use define a numerically stable version of `log(sum(exp(x)))`.
This function is in `scipy.misc`, but let's pretend we need to make our own version.

First, we need to import the usual numpy wrapper, plus the `primitive` class:

```python
import autograd.numpy as np
from autograd.core import primitive
```

Next, we define our function using standard Python, but using `@primitive` as a decorator:
```python
@primitive
def logsumexp(x):
"""Numerically stable log(sum(exp(x)))"""
max_x = np.max(x)
return max_x + np.log(np.sum(np.exp(x - max_x)))
```

`@primitive` tells autograd not to look inside this function, but instead
to treat it as a black box, whose gradient can be specified later.
Functions with this decorator can contain anything that Python knows
how to execute.

Next, we write a function that specifies the gradient of `logsumexp`.
In autograd, gradients are specified in a slightly roundabout way: through a function that returns a closure that evaluates the gradient.
This allows the gradient to dependon both the input to the original function (`x`), and the output of the original function (`ans`).

```python
def make_grad_logsumexp(ans, x):
def gradient_product(g):
return np.full(x.shape, g) * np.exp(x - np.full(x.shape, ans))
return gradient_product
```

If you want to be able to take higher-order derivatives, then all the
code inside this function must be itself differentiable by autograd.

The function `gradient_product` multiplies g with the Jacobian of `logsumexp`.
(d_ans/d_x).
Because autograd uses reverse-mode differentiation, g contains
the gradient of the objective w.r.t. ans, the output of logsumexp.

Now we tell autograd about `logsumexp`'s gradient-making function:
```python
logsumexp.defgrad(make_grad_logsumexp)
```

Now we can use logsumexp() anywhere, including inside of a larger function that we want to differentiate:

```python
def example_func(y):
z = y**2
lse = logsumexp(z)
return np.sum(lse)

grad_of_example = grad(example_func)
print "Gradient: ", grad_of_example(npr.randn(4))
```

This example can be found as a Python script [here](../examples/define_gradient.py).

### What if we have an unknown number of arguments?

## Planned features

Autograd is still under active development.  We plan to support:
* GPU operations
* In-place array operations and assignment to arrays
