import platform

import nox

NIGHTLY_INDEX_URL = "https://pypi.anaconda.org/scientific-python-nightly-wheels/simple"
UV_NIGHTLY_ENV_VARS = {
    "UV_INDEX_URL": NIGHTLY_INDEX_URL,
    "UV_PRERELEASE": "allow",
    "UV_INDEX_STRATEGY": "first-index",
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
    session.run("pytest", "--cov=autograd", "--cov-report=xml", "--cov-append", *session.posargs)


@nox.session(name="lint", reuse_venv=True)
def ruff(session):
    """Lightning-fast linting for Python"""
    session.install("pre-commit", silent=False)
    session.run("pre-commit", "run", "--all-files", "--show-diff-on-failure")


@nox.session(name="nightly-tests", tags=["tests"])
def run_nightly_tests(session):
    """Run tests against nightly versions of dependencies"""
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
    session.run("pytest", "--cov=autograd", "--cov-report=xml", "--cov-append", *session.posargs)
