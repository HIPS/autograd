# Autograd  [![Test status](https://travis-ci.org/HIPS/autograd.svg?branch=master)](https://travis-ci.org/HIPS/autograd)


Autograd can automatically differentiate native Python and Numpy code. It can
handle a large subset of Python's features, including loops, ifs, recursion and
closures, and it can even take derivatives of derivatives of derivatives. It
uses reverse-mode differentiation (a.k.a. backpropagation), which means it can
efficiently take gradients of scalar-valued functions with respect to
array-valued arguments. The main intended application is gradient-based
optimization. For more information, check out the [tutorial](docs/tutorial.md)
and the [examples directory](examples/).

Example use:

```python
>>> import autograd.numpy as np  # Thinly-wrapped numpy
>>> from autograd import grad    # The only autograd function you may ever need
>>>
>>> def tanh(x):                 # Define a function
...     y = np.exp(-x)
...     return (1.0 - y)  / ( 1.0 + y)
... 
>>> grad_tanh = grad(tanh)       # Obtain its gradient function
>>> grad_tanh(1.0)               # Evaluate the gradient at x = 1.0
0.39322386648296376
>>> (tanh(1.0001) - tanh(0.9999)) / 0.0002  # Compare to finite differences
0.39322386636453377
```

We can continue to differentiate as many times as we like:

```python
>>> grad_tanh_2 = grad(grad_tanh)           # 2nd derivative
>>> grad_tanh_3 = grad(grad_tanh_2)         # 3rd derivative
>>> grad_tanh_4 = grad(grad_tanh_3)         # etc.
>>> grad_tanh_5 = grad(grad_tanh_4)
>>> grad_tanh_6 = grad(grad_tanh_5)
>>>
>>> import matplotlib.pyplot as plt
>>> x = np.linspace(-7, 7, 200)
>>> plt.plot(x, map(tanh, x),
...          x, map(grad_tanh, x),
...          x, map(grad_tanh_2, x),
...          x, map(grad_tanh_3, x),
...          x, map(grad_tanh_4, x),
...          x, map(grad_tanh_5, x),
...          x, map(grad_tanh_6, x))
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

## How to install

Just run `pip install autograd`

## Authors

Autograd was written by [Dougal Maclaurin](mailto:maclaurin@physics.harvard.edu),
[David Duvenaud](http://mlg.eng.cam.ac.uk/duvenaud/)
and [Matt Johnson](http://www.mit.edu/~mattjj/),
and we're actively
developing it. Please feel free to submit any bugs or feature requests.
We'd also love to hear about your experiences with autograd in general.
Drop us an email!

We want to thank Jasper Snoek and the rest of the HIPS group (led by Prof. Ryan
P. Adams) for helpful contributions and advice; Barak Pearlmutter for
foundational work on automatic differentiation and for guidance on our
implementation; and Analog Devices Inc. (Lyric Labs) and Samsung Advanced Institute
of Technology for their generous support.
