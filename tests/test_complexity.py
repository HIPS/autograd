import time
from autograd import grad
import autograd.numpy as np

def timefunction(f):
    t = time.time()
    f()
    return time.time() - t

def assert_linear_time(f):
    t1 = timefunction(lambda: f(1))
    t3 = timefunction(lambda: f(3))
    assert 2 * t1 < t3 < 4 * t1, "f(1) takes{}, f(3) takes {}".format(t1, t3)

def test_array_creation():
    def fun(x, N):
        arr = [x for i in range(N)]
        return np.sum(np.array(arr))

    assert_linear_time(lambda N: grad(fun)(1.0, 3000*N))

def test_array_indexing():
    def fun(x):
        return sum([x[i] for i in range(len(x))])
    assert_linear_time(lambda N: grad(fun)(np.zeros(500*N)))

def test_list_indexing():
    def fun(x):
        return sum([x[i] for i in range(len(x))])
    assert_linear_time(lambda N: grad(fun)([0.0 for i in range(500*N)]))
