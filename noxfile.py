import platform

import nox

NIGHTLY_INDEX_URL = "https://pypi.anaconda.org/scientific-python-nightly-wheels/simple"
UV_NIGHTLY_ENV_VARS = {
    "UV_INDEX_URL": NIGHTLY_INDEX_URL,
    "UV_PRERELEASE": "allow",
    "UV_INDEX_STRATEGY": "first-index",
    "UV_NO_CACHE": "true",
}

nox.needs_version = ">=2024.4.15"
nox.options.default_venv_backend = "uv|virtualenv"
nox.options.reuse_existing_virtualenvs = False
nox.options.error_on_external_run = True
# nox.options.sessions = ["lint", "validate-package", "tests"]
nox.options.sessions = ["tests"]


@nox.session(name="validate-package")
def check(session):
    """Build source distribution, wheel, and check their metadata"""
    session.install("build", "twine", silent=False)
    session.run("python", "-m", "build")
    session.run("twine", "check", "--strict", "dist/*")


@nox.session(name="tests", tags=["tests"])
def run_tests(session):
    """Run unit tests and generate a coverage report"""
    # SciPy doesn't have wheels on PyPy
    if platform.python_implementation() == "PyPy":
        session.install("-e", ".[test]", silent=False)
    else:
        session.install("-e", ".[test,scipy]", silent=False)
    session.run(
        "pytest", "-n", "auto", "--cov=autograd", "--cov-report=xml", "--cov-append", *session.posargs
    )


@nox.session(name="lint", reuse_venv=True)
def ruff(session):
    """Lightning-fast linting for Python"""
    session.install("pre-commit", silent=False)
    session.run("pre-commit", "run", "--all-files", "--show-diff-on-failure")


@nox.session(name="nightly-tests", tags=["tests"])
def run_nightly_tests(session):
    """Run tests against nightly versions of dependencies"""
    session.run("python", "-VV")
    session.install("-e", ".[test]", silent=False)
    # SciPy doesn't have wheels on PyPy
    if platform.python_implementation() == "PyPy":
        session.install(
            "numpy", "--upgrade", "--only-binary", ":all:", silent=False, env=UV_NIGHTLY_ENV_VARS
        )
    else:
        session.install(
            "numpy", "scipy", "--upgrade", "--only-binary", ":all:", silent=False, env=UV_NIGHTLY_ENV_VARS
        )
    session.run(
        "pytest", "-n", "auto", "--cov=autograd", "--cov-report=xml", "--cov-append", *session.posargs
    )


# Wheels for NumPy and SciPy are available as nightly builds, so we test
# against them on Python 3.13t, which is the only version that supports
# free-threaded Python. This session is similar to the "nightly-tests"
# session, but it uses a free-threaded Python interpreter. Also, we don't
# the "test" extra but install the test dependencies manually.
#
# When the PYTHON_GIL environment variable is set to 0, we enforce that
# extension modules that haven't declared themselves as safe to not rely
# on the GIL are run with the GIL disabled.
@nox.session(name="free-threading", python=["3.13t"])
def run_with_free_threaded_python(session):
    """Run tests with free threaded Python (no-GIL)"""
    session.run("python", "-VV")
    session.install("-e", ".", silent=False)
    session.install("pytest", silent=False)

    # SciPy doesn't have wheels on PyPy
    if platform.python_implementation() == "PyPy":
        session.install(
            "numpy", "--upgrade", "--only-binary", ":all:", silent=False, env=UV_NIGHTLY_ENV_VARS
        )
    else:
        session.install(
            "numpy", "scipy", "--upgrade", "--only-binary", ":all:", silent=False, env=UV_NIGHTLY_ENV_VARS
        )
    session.run(
        "pytest",
        *session.posargs,
        env={"PYTHON_GIL": "0"},
    )


@nox.session(name="free-threading-pytest-run-parallel", python=["3.13t"])
def run_pytest_run_in_parallel_plugin(session):
    """Run stress tests with free threaded Python (no-GIL) using the pytest-run-in-parallel plugin"""
    session.run("python", "-VV")
    session.install("-e", ".", silent=False)
    session.install("pytest", "pytest-run-in-parallel", silent=False)

    # SciPy doesn't have wheels on PyPy
    if platform.python_implementation() == "PyPy":
        session.install(
            "numpy", "--upgrade", "--only-binary", ":all:", silent=False, env=UV_NIGHTLY_ENV_VARS
        )
    else:
        session.install(
            "numpy", "scipy", "--upgrade", "--only-binary", ":all:", silent=False, env=UV_NIGHTLY_ENV_VARS
        )
    session.run(
        "pytest" * session.posargs,
        env={"PYTHON_GIL": "0"},
    )


@nox.session(name="free-threading-pytest-freethreaded", python=["3.13t"])
def run_pytest_freethreaded(session):
    """Run stress tests with free threaded Python (no-GIL) using the pytest-freethreaded plugin"""
    session.run("python", "-VV")
    session.install("-e", ".", silent=False)
    session.install("pytest", "pytest-freethreaded", silent=False)

    # SciPy doesn't have wheels on PyPy
    if platform.python_implementation() == "PyPy":
        session.install(
            "numpy", "--upgrade", "--only-binary", ":all:", silent=False, env=UV_NIGHTLY_ENV_VARS
        )
    else:
        session.install(
            "numpy", "scipy", "--upgrade", "--only-binary", ":all:", silent=False, env=UV_NIGHTLY_ENV_VARS
        )
    session.run(
        "pytest",
        *session.posargs,
        env={"PYTHON_GIL": "0"},
    )
