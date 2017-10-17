import autograd.numpy as np
from autograd import grad, elementwise_grad
from autograd.misc.fixed_points import fixed_point
npr = np.random.RandomState(0)

def newton_sqrt_iter(a):
    return lambda x: 0.5 * (x + a / x)

def grad_descent_sqrt_iter(a):
    return lambda x: x - 0.05 * (x**2 - a)

def sqrt(a, guess=10.):
    # return fixed_point(newton_sqrt_iter, a, guess, distance, 1e-4)
    return fixed_point(grad_descent_sqrt_iter, a, guess, distance, 1e-4)

def distance(x, y): return np.abs(x - y)

print np.sqrt(2.)
print sqrt(2.)
print
print grad(np.sqrt)(2.)
print grad(sqrt)(2.)
print
print grad(grad(np.sqrt))(2.)
print grad(grad(sqrt))(2.)
print

N = 2
square = lambda X: np.dot(X, X.T)
A = square(npr.randn(N, N)) + np.eye(N)
b = npr.randn(N)

def make_quadratic(A, b):
    def quadratic(x):
        return 0.5 * np.dot(x, np.dot(A, x)) - np.dot(b, x)
    return quadratic

def grad_descent_quadratic(b):
    quadratic = make_quadratic(A, b)
    return lambda x: x - 0.05 * grad(quadratic)(x)

def minimize(b, guess=np.zeros(N)):
    return fixed_point(grad_descent_quadratic, b, guess, euclid, 1e-5)

def euclid(x, y):
    diff = x - y
    return np.sqrt(np.dot(diff, diff))

print np.linalg.solve(A, b)
print minimize(b)
print
print elementwise_grad(np.linalg.solve, 1)(A, b)
print elementwise_grad(minimize)(b)
print
