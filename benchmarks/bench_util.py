import autograd.numpy.random as npr
import autograd.numpy as np
from autograd import grad

try:
    from autograd.misc.flatten import flatten
except ImportError:
    from autograd.util import flatten

def time_flatten():
    val = {'k':  npr.random((4, 4)),
           'k2': npr.random((3, 3)),
           'k3': 3.0,
           'k4': [1.0, 4.0, 7.0, 9.0],
           'k5': np.array([4., 5., 6.]),
           'k6': np.array([[7., 8.], [9., 10.]])}

    vect, unflatten = flatten(val)
    val_recovered = unflatten(vect)
    vect_2, _ = flatten(val_recovered)

# def time_vspace_flatten():
#     val = {'k':  npr.random((4, 4)),
#            'k2': npr.random((3, 3)),
#            'k3': 3.0,
#            'k4': [1.0, 4.0, 7.0, 9.0],
#            'k5': np.array([4., 5., 6.]),
#            'k6': np.array([[7., 8.], [9., 10.]])}

#     vspace_flatten(val)

def time_grad_flatten():
    val = {'k':  npr.random((4, 4)),
           'k2': npr.random((3, 3)),
           'k3': 3.0,
           'k4': [1.0, 4.0, 7.0, 9.0],
           'k5': np.array([4., 5., 6.]),
           'k6': np.array([[7., 8.], [9., 10.]])}

    vect, unflatten = flatten(val)
    def fun(vec):
        v = unflatten(vec)
        return np.sum(v['k5']) + np.sum(v['k6'])

    grad(fun)(vect)
