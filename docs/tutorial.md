# Autograd tutorial

## Motivation

Imagine you want to test out a new machine learning model for your data. This
usually means coming up with some loss function to capture how well your model
fits the data and optimizing that loss with respect to the model parameters. If
there are many model parameters (neural nets can have millions) then you need
gradients. You then have two options: derive and code them up yourself, or
implement your model using the syntactic and semantic constraints of a system
like [Theano](http://deeplearning.net/software/theano/) or
[TensorFlow](https://github.com/tensorflow/tensorflow).

We want to provide a third way: just write down the loss function using a
standard numerical library like Numpy, and Autograd will give you its gradient.

## How to use Autograd

Autograd's `grad` function takes in a function, and gives you a function that computes its derivative.
Your function must have a scalar-valued output (i.e. a float).
This covers the common case when you want to use gradients to optimize something.

Autograd works on ordinary Python and Numpy code containing all the usual control structures, including `while` loops, `if` statements, and closures.  Here's a simple example of using an open-ended loop to compute the sine function:

```python
import autograd.numpy as np   # Thinly-wrapped version of Numpy
from autograd import grad

def taylor_sine(x):  # Taylor approximation to sine function
    ans = currterm = x
    i = 0
    while np.abs(currterm) > 0.001:
        currterm = -currterm * x**2 / ((2 * i + 3) * (2 * i + 2))
        ans = ans + currterm
        i += 1
    return ans

grad_sine = grad(taylor_sine)
print "Gradient of sin(pi) is", grad_sine(np.pi)
```

## Complete example: logistic regression

A common use case for automatic differentiation is to train a probabilistic model.
Here we present a very simple (but complete) example of specifying and training
a logistic regression model for binary classification:

```python
import autograd.numpy as np
from autograd import grad

def sigmoid(x):
    return 0.5 * (np.tanh(x / 2.) + 1)

def logistic_predictions(weights, inputs):
    # Outputs probability of a label being true according to logistic model.
    return sigmoid(np.dot(inputs, weights))

def training_loss(weights):
    # Training loss is the negative log-likelihood of the training labels.
    preds = logistic_predictions(weights, inputs)
    label_probabilities = preds * targets + (1 - preds) * (1 - targets)
    return -np.sum(np.log(label_probabilities))

# Build a toy dataset.
inputs = np.array([[0.52, 1.12,  0.77],
                   [0.88, -1.08, 0.15],
                   [0.52, 0.06, -1.30],
                   [0.74, -2.49, 1.39]])
targets = np.array([True, True, False, True])

# Define a function that returns gradients of training loss using Autograd.
training_gradient_fun = grad(training_loss)

# Optimize weights using gradient descent.
weights = np.array([0.0, 0.0, 0.0])
print("Initial loss:", training_loss(weights))
for i in range(100):
    weights -= training_gradient_fun(weights) * 0.01

print("Trained loss:", training_loss(weights))
```

Python syntax is pretty good for specifying probabilistic models.  The biggest
win is that it becomes a lot easier to modify a model and rapidly iterate.

For more complex examples, see our [examples directory](../examples/), which includes:
* [a simple neural net](../examples/neural_net.py)
* [a convolutional neural net](../examples/convnet.py)
* [a recurrent neural net](../examples/rnn.py)
* [a long short-term memory (LSTM)](../examples/lstm.py)
* [backpropagating through a fluid simulation](../examples/fluidsim/fluidsim.py)


## What's going on under the hood?

To compute the gradient, Autograd first has to record every transformation that was applied to the input as it was turned into the output of your function.
To do this, Autograd wraps functions (using the function `primitive`) so that when they're called, they add themselves to a list of operations performed.
Autograd's core has a table mapping these wrapped primitives to their corresponding gradient functions (or, more precisely, their vector-Jacobian product functions).
To flag the variables we're taking the gradient with respect to, we wrap them using the `Box` class.
You should never have to think about the `Box` class, but you might notice it when printing out debugging info.

After the function is evaluated, Autograd has a graph specifying all operations that were performed on the inputs with respect to which we want to differentiate.
This is the computational graph of the function evaluation.
To compute the derivative, we simply apply the rules of differentiation to each node in the graph.

### Reverse mode differentiation

Given a function made up of several nested function calls, there are several ways to compute its derivative.

For example, given L(x) = F(G(H(x))), the chain rule says that its gradient is dL/dx = dF/dG * dG/dH * dH/dx.  If we evaluate this product from right-to-left: (dF/dG * (dG/dH * dH/dx)), the same order as the computations themselves were performed, this is called forward-mode differentiation.
If we evaluate this product from left-to-right: (dF/dG * dG/dH) * dH/dx)), the reverse order as the computations themselves were performed, this is called reverse-mode differentiation.

Compared to finite differences or forward-mode, reverse-mode differentiation is by far the more practical method for differentiating functions that take in a large vector and output a single number.
In the machine learning community, reverse-mode differentiation is known as 'backpropagation', since the gradients propagate backwards through the function.
It's particularly nice since you don't need to instantiate the intermediate Jacobian matrices explicitly, and instead only rely on applying a sequence of matrix-free vector-Jacobian product functions (VJPs).
Because Autograd supports higher derivatives as well, Hessian-vector products (a form of second-derivative) are also available and efficient to compute.

### How can you support ifs, while loops and recursion?

Some autodiff packages (such as [TensorFlow](https://github.com/tensorflow/tensorflow)) work by having you specify a graph of the computation that your function performs, including all the control flow (such as if and for loops), and then turn that graph into another one that computes gradients.
This has some benefits (such as allowing compile-time optimizations), but it requires you to express control flow in a limited mini-language that those packages know how to handle.  (For example, the `tf.while` and `tf.cond` operations in TensorFlow.)

In contrast, Autograd doesn't have to know about any ifs, branches, loops or recursion that were used to decide which operations were called.  To compute the gradient of a particular input, one only needs to know which continuous transforms were applied to that particular input, not which other transforms might have been applied.
Since Autograd keeps track of the relevant operations on each function call separately, it's not a problem that all the Python control flow operations are invisible to Autograd.  In fact, it greatly simplifies the implementation.


## What can Autograd differentiate?

The main constraint is that any function that operates on a `Box` is marked as `primitive`, and has its gradient implemented.
This is taken care of for most functions in the Numpy library, and it's easy to write your own gradients.

The input can be a scalar, complex number, vector, tuple, a tuple of vectors, a tuple of tuples, etc.

When using the `grad` function, the output must be a scalar, but the functions `elementwise_grad` and `jacobian` allow gradients of vectors.


## Supported and unsupported parts of numpy/scipy

Numpy has [a lot of features](http://docs.scipy.org/doc/numpy/reference/). We've done our best to support most of them. So far, we've implemented gradients for:
* most of the [mathematical operations](../autograd/numpy/numpy_vjps.py)
* most of the [array and matrix manipulation routines](../autograd/numpy/numpy_vjps.py)
* some [linear algebra](../autograd/numpy/linalg.py) functions
* most of the [fast fourier transform](../autograd/numpy/fft.py) routines
* full support for complex numbers
* [N-dimensional convolutions](../autograd/scipy/signal.py)
* Some scipy routines, including [`scipy.stats.norm`](../autograd/scipy/stats/norm.py)

Some things remain to be implemented. For example, we support indexing (`x = A[i, j, :]`) but not assignment (`A[i,j] = x`) in arrays that are being differentiated with respect to.
Assignment is hard to support because it requires keeping copies of the overwritten data, and so even when you write code that looks like it's performing assignment, the system would have to be making copies behind the scenes, often defeating the purpose of in-place operations.

Similarly, we don't support the syntax `A.dot(B)`; use the equivalent `np.dot(A, B)` instead.
The reason we don't support the first way is that subclassing `ndarray` raises a host of issues.
As another consequence of not subclassing `ndarray`, some subclass checks can break, like `isinstance(x, np.ndarray)` can return `False`.
However, those `isinstance` checks will work if you instead use Autograd's provided one, writing `from autograd.builtins import isinstance`.

In-place modification of arrays not being differentiated with respect to (for example, `A[i] = x` or `A += B`) won't raise an error, but be careful.
It's easy to accidentally change something without Autograd knowing about it.
This can be a problem because Autograd keeps references to variables used in the forward pass if they will be needed on the reverse pass.
Making copies would be too slow.

Lists and dicts can be used freely - like control flow, Autograd usually doesn't even need to know about them.
The exception is passing in a list to a primitive function, such as `autograd.numpy.sum`.
This requires special care, since the list contents need to be examined for boxes.
We do support passing lists to `autograd.numpy.array` and `autograd.numpy.concatenate`, but in other cases, you may need to explicitly construct an array using `autograd.numpy.array` before passing a list or tuple argument into a primitive.
An alternative is to use the `list`, `dict`, and `tuple` classes in `autograd.builtins`, which should work just like the Python builtins while also ensuring boxes don't get hidden inside those containers.
Remember, these issues typically only come up when you're passing a `list` or `tuple` to a primitive function; when passing around lists or tuples in your own (non-primitive) functions, you can put boxed values inside lists, tuples, or dicts without having to worry about it.

#### TL;DR: Do use
* [Most](../autograd/numpy/numpy_vjps.py) of numpy's functions
* [Most](../autograd/numpy/numpy_boxes.py) numpy.ndarray methods
* [Some](../autograd/scipy/) scipy functions
* Indexing and slicing of arrays `x = A[3, :, 2:4]`
* Explicit array creation from lists `A = np.array([x, y])`

#### Don't use
* Assignment to arrays `A[0,0] = x`
* Implicit casting of lists to arrays `A = np.sum([x, y])`, use `A = np.sum(np.array([x, y]))` instead.
* `A.dot(B)` notation (use `np.dot(A, B)` instead)
* In-place operations (such as `a += b`, use `a = a + b` instead)
* Some isinstance checks, like `isinstance(x, np.ndarray)` or `isinstance(x, tuple)`, without first doing `from autograd.builtins import isinstance, tuple`.

Luckily, it's easy to check gradients numerically if you're worried that something's wrong.

## Extend Autograd by defining your own primitives

What if Autograd doesn't support a function you need to take the gradient of?
This can happen if your code depends on external library calls or C code.
It can sometimes even be a good idea to provide the gradient of a pure Python function for speed or numerical stability.

For example, let's add the gradient of a numerically stable version of `log(sum(exp(x)))`.
This function is included in `scipy.special` and already supported, but let's make our own version.

Next, we define our function using standard Python, using `@primitive` as a decorator:

```python
import autograd.numpy as np
from autograd.extend import primitive, defvjp

@primitive
def logsumexp(x):
    """Numerically stable log(sum(exp(x)))"""
    max_x = np.max(x)
    return max_x + np.log(np.sum(np.exp(x - max_x)))
```

`@primitive` tells Autograd not to look inside the function, but instead to treat it as a black box whose gradient can be specified later.
Functions with this decorator can contain anything that Python knows how to execute, including calls to other languages.

Next, we write a function that specifies the gradient of the primitive `logsumexp`:

```python
def logsumexp_vjp(ans, x):
    x_shape = x.shape
    return lambda g: np.full(x_shape, g) * np.exp(x - np.full(x_shape, ans))
```

`logsumexp_vjp` returns a vector-Jacobian product (VJP) operator, which is a function that right-multiplies its argument `g` by the Jacobian matrix of `logsumexp` (without explicitly forming the matrix's coefficients).
`g` will be the gradient of the final objective with respect to `ans` (the output of `logsumexp`).
The calculation can depend on both the input (`x`) and the output (`ans`) of the original function.
If you want to be able to take higher-order derivatives, then the code inside the VJP function must be itself differentiable by Autograd, which usually just means you write it in terms of other primitives which themselves have VJPs (like Numpy functions).

The final step is to tell Autograd about `logsumexp`'s vector-Jacobian product function:
```python
defvjp(logsumexp, logsumexp_vjp)
```

Now we can use `logsumexp` anywhere, including inside of a larger function that we want to differentiate:

```python
from autograd import grad

def example_func(y):
    z = y**2
    lse = logsumexp(z)
    return np.sum(lse)

grad_of_example = grad(example_func)
print "Gradient: ", grad_of_example(np.array([1.5, 6.7, 1e-10])
```

This example can be found as a Python script [here](../examples/define_gradient.py).

## Complex numbers

Autograd supports complex arrays and scalars using a convention described as follows.
Consider a complex-to-complex function, `f`,
expressed in terms of real-to-real components, `u` and `v`:

```python
def f(z):
    x, y = real(z), imag(z)
    return u(x, y) + v(x, y) * 1j
```

We define `grad` of `f` as

```python
def grad_f(z):
    x, y = real(z), imag(z)
    return grad(u, 0)(x, y) - i * grad(u, 1)(x, y)
```

(The second argument of `grad` specifies which argument we're differentiating with respect to.)
So we throw out v, the imaginary part of f, entirely.

Our convention covers three important cases:
  * If `f` is holomorphic, we get the usual complex derivative
    (since `grad(u, 0) == grad(v, 1)` and `grad(u, 1) == - grad(v, 0)`).
  * If `f` is a real-valued loss function of a complex parameter, `x`,
    we get a result that we can use in a gradient-based optimizer,
    by taking steps in the direction of the complex conjugate of `grad(f)(x)`.
  * If `f` is a real-to-real function that happens to use complex primitives internally,
    some of which must necessarily be non-holomorphic
    (maybe you use FFTs to implement convolutions for example)
    then we get the same result that a purely real implementation would have given.

Our convention doesn't handle the case where `f` is a non-holomorphic function
and you're interested in all of du/dx, du/dy, dv/dx and dv/dy.
But then the answer would have to contain four real values
and there would be no way to express it as a single complex number.

We define primitive vector-Jacobian products of complex functions like this

```python
def f_vjp(g, z):
    z_x, z_y = real(z), imag(z)
    g_x, g_y = real(g), imag(g)
    return (       g_x * grad(u, 0)(x, y)
             - i * g_x * grad(u, 1)(x, y)
             -     g_y * grad(v, 0)(x, y)
             + i * g_y * grad(v, 1)(x, y))
```

For holomorphic primitives, this is just the regular complex derivative multiplied by `g`,
so most simple math primitives don't need to be changed from their real implementations.
For non-holomorphic primitives, it preserves all four real partial derivatives as if we
were treating complex numbers as real 2-tuples
(though it throws a couple of negative signs in there).
Chapter 4 of [Dougal's PhD thesis](https://dougalmaclaurin.com/phd-thesis.pdf)
goes into a bit more detail about how we define the primitive vector-Jacobian products.

## Autograd Lecture
For more information on automatic differentiation, autograd's implementation, and advanced automatic differentiation techniques, see a [talk by Matt at the Deep Learning Summer School, Montreal 2017](http://videolectures.net/deeplearning2017_johnson_automatic_differentiation/).

## Support

Autograd was written by
[Dougal Maclaurin](https://dougalmaclaurin.com),
[David Duvenaud](http://mlg.eng.cam.ac.uk/duvenaud/), and
[Matthew Johnson](http://www.mit.edu/~mattjj/)
and we're actively developing it. Please
feel free to submit any bugs or feature requests. We'd also love to hear about
your experiences with Autograd in general. Drop us an email!
