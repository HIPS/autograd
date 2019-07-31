__Note: Autograd is still being maintained but is no longer actively developed.
The main developers (Dougal Maclaurin, David Duvenaud, Matt Johnson, and Jamie
Townsend) are now working on [JAX](https://github.com/google/jax), with Dougal and Matt
working on it full-time. JAX combines a new version of Autograd with extra
features such as jit compilation.__

# Autograd  [![Test status](https://travis-ci.org/HIPS/autograd.svg?branch=master)](https://travis-ci.org/HIPS/autograd) [![asv](http://img.shields.io/badge/benchmarked%20by-asv-green.svg?style=flat)](#)

Autograd can automatically differentiate native Python and Numpy code. It can
handle a large subset of Python's features, including loops, ifs, recursion and
closures, and it can even take derivatives of derivatives of derivatives. It
supports reverse-mode differentiation (a.k.a. backpropagation), which means it
can efficiently take gradients of scalar-valued functions with respect to
array-valued arguments, as well as forward-mode differentiation, and the two can
be composed arbitrarily. The main intended application of Autograd is
gradient-based optimization. For more information, check out the
[tutorial](docs/tutorial.md) and the [examples directory](examples/).

Example use:

```python
>>> import autograd.numpy as np  # Thinly-wrapped numpy
>>> from autograd import grad    # The only autograd function you may ever need
>>>
>>> def tanh(x):                 # Define a function
...     y = np.exp(-2.0 * x)
...     return (1.0 - y) / (1.0 + y)
...
>>> grad_tanh = grad(tanh)       # Obtain its gradient function
>>> grad_tanh(1.0)               # Evaluate the gradient at x = 1.0
0.41997434161402603
>>> (tanh(1.0001) - tanh(0.9999)) / 0.0002  # Compare to finite differences
0.41997434264973155
```

We can continue to differentiate as many times as we like, and use numpy's
vectorization of scalar-valued functions across many different input values:

```python
>>> from autograd import elementwise_grad as egrad  # for functions that vectorize over inputs
>>> import matplotlib.pyplot as plt
>>> x = np.linspace(-7, 7, 200)
>>> plt.plot(x, tanh(x),
...          x, egrad(tanh)(x),                                     # first  derivative
...          x, egrad(egrad(tanh))(x),                              # second derivative
...          x, egrad(egrad(egrad(tanh)))(x),                       # third  derivative
...          x, egrad(egrad(egrad(egrad(tanh))))(x),                # fourth derivative
...          x, egrad(egrad(egrad(egrad(egrad(tanh)))))(x),         # fifth  derivative
...          x, egrad(egrad(egrad(egrad(egrad(egrad(tanh))))))(x))  # sixth  derivative
>>> plt.show()
```

<img src="examples/tanh.png" width="600">

See the [tanh example file](examples/tanh.py) for the code.

## Documentation

You can find a tutorial [here.](docs/tutorial.md)

## End-to-end examples

* [Simple neural net](examples/neural_net.py)
* [Convolutional neural net](examples/convnet.py)
* [Recurrent neural net](examples/rnn.py)
* [LSTM](examples/lstm.py)
* [Neural Turing Machine](https://github.com/DoctorTeeth/diffmem/blob/512aadeefd6dbafc1bdd253a64b6be192a435dc3/ntm/ntm.py)
* [Backpropagating through a fluid simulation](examples/fluidsim/fluidsim.py)

<img src="examples/fluidsim/animated.gif" width="400">

* [Variational inference in Bayesian neural network](examples/bayesian_neural_net.py)
* [Gaussian process regression](examples/gaussian_process.py)
* [Sampyl, a pure Python MCMC package with HMC and NUTS](https://github.com/mcleonard/sampyl)

## How to install

Just run `pip install autograd`

## Authors

Autograd was written by [Dougal Maclaurin](https://dougalmaclaurin.com),
[David Duvenaud](https://www.cs.toronto.edu/~duvenaud/),
[Matt Johnson](http://people.csail.mit.edu/mattjj/),
[Jamie Townsend](https://github.com/j-towns)
and many other contributors. The package is currently still being maintained,
but is no longer actively developed. Please feel free to submit any bugs or
feature requests. We'd also love to hear about your experiences with autograd
in general. Drop us an email!

We want to thank Jasper Snoek and the rest of the HIPS group (led by Prof. Ryan
P. Adams) for helpful contributions and advice; Barak Pearlmutter for
foundational work on automatic differentiation and for guidance on our
implementation; and Analog Devices Inc. (Lyric Labs) and Samsung Advanced Institute
of Technology for their generous support.
