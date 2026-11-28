import nox

NIGHTLY_INDEX_URL = "https://pypi.anaconda.org/scientific-python-nightly-wheels/simple"
UV_NIGHTLY_ENV_VARS = {
    "UV_INDEX_URL": NIGHTLY_INDEX_URL,
    "UV_PRERELEASE": "allow",
    "UV_INDEX_STRATEGY": "first-index",
    "UV_NO_CONFIG": "1",
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
    session.run("uv", "build")
    session.run("twine", "check", "--strict", "dist/*")


@nox.session(name="tests", tags=["tests"])
def run_tests(session):
    """Run unit tests and generate a coverage report"""
    pyproject = nox.project.load_toml("pyproject.toml")
    session.install(*nox.project.dependency_groups(pyproject, "test"))
    session.install("-e", ".[scipy]", silent=False)
    session.run("pytest", "--cov=autograd", "--cov-report=xml", "--cov-append", *session.posargs)


@nox.session(name="oldest-tests", venv_backend="uv", tags=["tests"])
def run_oldest_tests(session):
    """Run tests against the oldest supported versions of NumPy and SciPy"""
    pyproject = nox.project.load_toml("pyproject.toml")
    # Pin our direct dependencies to their lower bounds
    constraints = f"{session.create_tmp()}/oldest.txt"
    session.run(
        "uv",
        "pip",
        "compile",
        "pyproject.toml",
        "--extra",
        "scipy",
        "--resolution",
        "lowest-direct",
        "--output-file",
        constraints,
        silent=True,
    )
    session.install(
        "-e",
        ".[scipy]",
        *nox.project.dependency_groups(pyproject, "test"),
        "--constraint",
        constraints,
        silent=False,
    )
    session.run("pytest", "--cov=autograd", "--cov-report=xml", "--cov-append", *session.posargs)


@nox.session(name="lint", reuse_venv=True)
def ruff(session):
    """Lightning-fast linting for Python"""
    session.install("prek", silent=False)
    session.run("prek", "-a")


@nox.session(name="nightly-tests", tags=["tests"])
def run_nightly_tests(session):
    """Run tests against nightly versions of dependencies"""
    session.install("-e.", silent=False)
    pyproject = nox.project.load_toml("pyproject.toml")
    session.install(*nox.project.dependency_groups(pyproject, "test"))
    session.install(
        "numpy",
        "scipy",
        "--upgrade",
        "--only-binary",
        ":all:",
        silent=False,
        env=UV_NIGHTLY_ENV_VARS,
    )
    session.run("pytest", "--cov=autograd", "--cov-report=xml", "--cov-append", *session.posargs)
