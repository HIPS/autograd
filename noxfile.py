import nox

nox.needs_version = ">=2024.4.15"
nox.options.default_venv_backend = "uv|virtualenv"
nox.options.reuse_existing_virtualenvs = True
nox.options.error_on_external_run = True
# nox.options.sessions = ["lint", "validate-package", "tests"]
nox.options.sessions = ["tests"]


@nox.session(name="validate-package")
def check(session):
    """Build source distribution, wheel, and check their metadata"""
    session.install("build", "twine")
    session.run('python', '-m', 'build')
    session.run('twine', 'check', '--strict', "dist/*")


@nox.session(name="tests")
def run_tests(session):
    """Run unit tests and generate a coverage report"""
    session.install("-e", ".[test,scipy]")
    session.run("pytest", "--cov=autograd", "--cov-report=xml", *session.posargs)


# TODO: Replace with pre-commit and pre-commit.ci once
# https://github.com/HIPS/autograd/pull/634 is merged
@nox.session(name="lint")
def ruff(session):
    """Lightning-fast linting for Python"""
    session.install("ruff")
    session.run('ruff', 'check', '.')
    session.notify("tests")
