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

## How to use autograd

Autograd's `grad` method takes in a function, and gives you a function that computes its derivative.
Your function must have a scalar-valued output (i.e. just a floating point number).
This covers the common case when you want to use gradients to optimize something.

* Grad of a function with loops, etc


## Complete example: logistic regression
* Vector inputs is where the action is
* Do a grad check


For more complex examples, see our [examples directory](../examples/), which includes:
* [a simple neural net](../examples/neural_net.py)
* [a convolutional neural net](../examples/convnet.py)
* [a rcurrent neural net](../examples/rnn.py)
* [a long short-term memory (LSTM)](../examples/lstm.py)
* [backpropagating through a fluid simulation](../examples/fluidsim/fluidsim.py) 



## What's going on under the hood?

To compute the gradient, autograd first has to record every transformation that was applied to the input as it was transformed into the output of your function. It does this by wrapping the input as a `node` class, and by enforcing that every continuous operation that depends on a node (such as `exp`) also outputs a node, and adds itself to the list of operations performed.

After the function is evaluated, autograd has a list of all operations that were performed, and which nodes they depended on.  This is the computational graph of the function.  To compute the derivative, we simply have to apply the rules of differentiation to each node in the graph.

### Reverse mode differentiation

Given a function made up of several nested function calls, there are several ways to compute its derivative.

For exammple, given L(x) = F(G(H(x))), the chain rule says that its gradient is dL/dx = dF/dG * dG/dH * dH/dx.  If we evaluate this product from right-to-left: (dF/dG * (dG/dH * dH/dx)), the same order as the computations themselves were performed, this is called forward-mode differentiation.
If we evaluate this product from left-to-right: (dF/dG * dG/dH) * dH/dx)), the same order as the computations themselves were performed, this is called forward-mode differentiation.

Compared to finite differences or forward-mode, reverse-mode differentiation is by far the most practical method for differentiating functions that take in a large vector and output a single number.
In the machine learning community, reverse-mode differentiation is known as 'backpropagation', since the gradients propogate backwards through the function.
It's particularly nice since you don't need to instantiate Jacobians, and Jacobian-vector products can often be computed efficiently as well.
Because autograd supports higher derivatives as well, Hessian-vector products (a form of second-derivative) are also available and efficiently computable.

### How can you support ifs, while loops and recursion?

Some autodiff packages (Theano, Kayak) work by having you specify a graph of the computation that your function performs, including all the control flow (such as if and for loops), then turn that graph into another one that computes gradients.
This has some benefits, but the price you pay is the need to express the control flow in a limited mini-language that those packages know how to handle.  (For example, the `scan()` operation in Theano.)

In contrast, autograd doesn't have to explicitly know about any ifs, branches, loops or recursion that were used to decide which operations were called.  To compute the gradient of a particular input, one only needs to know which continuous transforms were applied for that particular input, not which other transforms might have been applied.
Since autograd keeps track of the relevant operations on each function call separately, it's not a problem that all the Python control flow operations are invisible to autograd.  In fact, it greatly simplifies the implementation.


## What can autograd differentiate?

Autograd can differentiate through `if` statements, `for` and `while` loops, recursive function calls, and closures.  The main constraint is that any function that operates on a Node is marked as `primitive`, and has its gradient implemented.
This is taken care of for most functions in the Numpy library, and it's easy to write your own gradients.  Variables in your program will become nodes if they're continuous transformation of the input we're differentiating with respect to.

The input can be a vector, tuple, a tuple of vectors, a tuple of tuples, etc.

## Supported and unsupported parts of numpy/scipy

Numpy has [a lot of features](http://docs.scipy.org/doc/numpy/reference/). We've done our best to support most of them.  We've implemented gradients for:
* most of the [mathematical operations](../autograd/numpy/numpy_grads.py)
* most of the [array and matrix manipulation routines](../autograd/numpy/numpy_grads.py)
* some [linear algebra](../autograd/numpy/linalg.py) functions
* most of the [fast fourier transform](../autograd/numpy/fft.py) routines
* full support for complex numbers
* [N-dimensional convolutions](../autograd/scipy/signal.py)
* Some scipy routines, including [`scipy.stats.norm`](../autograd/scipy/stats/norm.py)

Some things remain to be implemented. For example, we support indexing: `x = A[i, j, :]`, but not assignment: `A[i,j] = x`.
Assignment is hard to support because it requires keeping copies of the overwritten data, but we plan to support this in the future.

Similarly, one particular way of calling binary array methods like ndarray.dot isn't supported - instead of calling `A.dot(B)`, use the equivalent `np.dot(A, B)`.  The reason we don't support the first way is that subclassing `ndarray` raises a host of other issues.

Assignment to regular arrays is supported, but be careful!  It's easy to accidentally change something without autograd knowing about it.  Luckily, it's easy to check gradients numerically if you're worried that something's wrong.

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

What if autograd doesn't support a function you need to take the gradient of?
This can happen if your code depends on external library calls or C code.
It can also be a good idea to provide the gradient of a pure Python function for speed or numerical stability.

For example, let's add the gradient of a numerically stable version of `log(sum(exp(x)))`.
This function is included in `scipy.misc` and already supported, but let's pretend we need to make our own version.

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
to treat it as a black box whose gradient can be specified later.
Functions with this decorator can contain anything that Python knows
how to execute, including calls to other languages.

Next, we write a function that specifies the gradient of `logsumexp`.
In autograd, gradients are specified in a slightly roundabout way: through a function that returns a closure that evaluates the gradient:

```python
def make_grad_logsumexp(ans, x):
    def gradient_product(g):
        return np.full(x.shape, g) * np.exp(x - np.full(x.shape, ans))
    return gradient_product
```
This allows the gradient to depend on both the input to the original function (`x`), and the output of the original function (`ans`).

If you want to be able to take higher-order derivatives, then all the
code inside this function must be itself differentiable by autograd.

The function `gradient_product` multiplies g with the Jacobian of `logsumexp`.
Because autograd uses reverse-mode differentiation, `g` contains
the gradient of the objective with respect to `ans` (the output of `logsumexp`).

The final step is to tell autograd about `logsumexp`'s gradient-making function:
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
