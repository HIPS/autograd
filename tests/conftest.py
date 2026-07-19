import pytest

import autograd.numpy.random as npr


@pytest.fixture
def rng():
    """A seeded RNG, private to each test.

    Under pytest-run-parallel, the hook below replaces this value with a
    fresh, identically seeded instance in every thread, so concurrent runs
    of a test see the same stream as a single-threaded run.
    """
    return npr.RandomState(42)


# optionalhook lets pytest ignore this hookimpl when pytest-run-parallel
# (which provides the hookspec) is not installed.
@pytest.hookimpl(optionalhook=True)
def pytest_run_parallel_get_thread_setups(n_workers):
    def fresh_rng(value, *, thread_index):
        return npr.RandomState(42)

    return {"rng": fresh_rng}
