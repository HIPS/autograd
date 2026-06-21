import threading

import numpy as onp

import autograd.numpy as np
from autograd import grad, jacobian


def run_threaded(worker, n_threads=8):
    barrier = threading.Barrier(n_threads)
    errors = []

    def target(i):
        try:
            barrier.wait()
            worker(i)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=target, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]


def test_concurrent_grad_distinct_inputs():
    """
    Each thread differentiates the same function at a different input and
    must recover its own gradient, with no cross-contamination of the trace.
    """
    fun = lambda x: np.sum(np.sin(x) ** 2 + np.exp(x))
    g = grad(fun)
    inputs = [onp.linspace(0.1, 1.0, 6) + i for i in range(8)]
    expected = [g(x) for x in inputs]  # serial reference

    results = [None] * len(inputs)

    def worker(i):
        for _ in range(50):
            results[i] = g(inputs[i])

    run_threaded(worker, n_threads=len(inputs))

    for i, exp in enumerate(expected):
        onp.testing.assert_allclose(results[i], exp, rtol=1e-12, atol=1e-12)


def test_concurrent_higher_order_grad():
    """
    Running higher-order grads nests traces, so doing so concurrently
    here is a direct test of per-thread trace-counter isolation.
    """
    fun = lambda x: np.sin(x)
    third = grad(grad(grad(fun)))
    inputs = [0.1 + 0.1 * i for i in range(8)]
    expected = [third(x) for x in inputs]

    results = [None] * len(inputs)

    def worker(i):
        for _ in range(50):
            results[i] = third(inputs[i])

    run_threaded(worker, n_threads=len(inputs))

    for i, exp in enumerate(expected):
        onp.testing.assert_allclose(results[i], exp, rtol=1e-12, atol=1e-12)


def test_concurrent_jacobian():
    """Reverse-mode jacobian under concurrency and distinct inputs per thread"""
    fun = lambda x: np.dot(x, x) * np.sin(x)
    jac = jacobian(fun)
    inputs = [onp.linspace(0.1, 1.0, 4) + i for i in range(8)]
    expected = [jac(x) for x in inputs]

    results = [None] * len(inputs)

    def worker(i):
        for _ in range(50):
            results[i] = jac(inputs[i])

    run_threaded(worker, n_threads=len(inputs))

    for i, exp in enumerate(expected):
        onp.testing.assert_allclose(results[i], exp, rtol=1e-12, atol=1e-12)
