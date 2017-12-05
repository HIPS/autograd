import time
import warnings
from autograd import grad, deriv
import autograd.numpy as np
from autograd.builtins import list as make_list

def timefunction(f):
    t = time.time()
    f()
    return time.time() - t

def assert_linear_time(f):
    t = timefunction(lambda: f(1))
    t10 = timefunction(lambda: f(10))
    assert t10 >  5 * t, "Too fast: f(1) takes {}, f(10) takes {}".format(t, t10)
    assert t10 < 20 * t, "Too slow: f(1) takes {}, f(10) takes {}".format(t, t10)
    if not (8 * t < t10 < 12 * t):
        warnings.warn("Borderline linearity. May fail on different hardware")

def test_array_creation():
    def fun(x, N):
        arr = [x for i in range(N)]
        return np.sum(np.array(arr))

    assert_linear_time(lambda N: grad(fun)(1.0, 200*N))

def test_array_indexing():
    def fun(x):
        return sum([x[i] for i in range(len(x))])
    assert_linear_time(lambda N: grad(fun)(np.zeros(200*N)))

def test_list_indexing():
    def fun(x):
        return sum([x[i] for i in range(len(x))])
    assert_linear_time(lambda N: grad(fun)([0.0 for i in range(50*N)]))

def test_list_creation():
    def fun(x, N):
        return make_list(*[x for _ in range(N)])
    assert_linear_time(lambda N: deriv(fun)(0.0, 20*N))

# This fails. Need to figure out why
def test_array_creation_fwd():
    def fun(x, N):
        arr = [x for i in range(N)]
        return np.sum(np.array(arr))

    assert_linear_time(lambda N: deriv(fun)(1.0, 400*N))
