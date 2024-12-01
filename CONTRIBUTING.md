# Contributing

Use [Nox](https://nox.thea.codes/en/stable/) to run tests and linting, e.g.,

```shell
pip install nox
```

`nox` will run all checks in an isolated virtual environment with Autograd and its dependencies, including its optional dependencies, installed.

## Run tests, linting, packaging checks

| Command                   | Description                                                                                                                                                     |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `nox --list`              | Lists all available Nox sessions, including selected ones                                                                                                       |
| `nox -s lint`             | Runs code style checks with pre-commit and pre-commit hooks as listed in `.pre-commit-config.yaml`. Accepts posargs to pass additional arguments to the linter. |
| `nox -s tests`            | Runs tests with your default Python interpreter. Accepts posargs to pass additional arguments and configuration to `pytest`.                                    |
| `nox -s nightly-tests`    | Similar to `nox -s tests`, except that it runs tests with nightly versions of dependencies (NumPy, SciPy, etc.).                                                |
| `nox -s validate-package` | Builds a source distribution and a wheel using `pypa/build` and checks the package with `twine` in strict mode.                                                 |
| `nox`                     | Runs all selected sessions, as listed in `nox.options.sessions` in `noxfile.py`.                                                                                |

Additionally, `nox` supports tags to run specific sessions, e.g., `nox --tags tests` runs all sessions tagged with `tests`.

Make sure all tests pass before you push your changes to GitHub.
GH Actions will run the tests across all supported Python versions.

## Using positional arguments (reformat, upload package, help)

You can use additional arguments for the tools (`pytest`, `pre-commit`, etc.) called by Nox by
separating them from the Nox arguments by a double-hyphen `--`, e.g.,

- `nox -s tests -- --tests/test_tuple.py` runs just the tests listed `tests/test_tuple.py`.
- `nox -s lint -- --fix` runs the linter with the `--fix` flag.
- and so on.
