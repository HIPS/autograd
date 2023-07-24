# Contributing

Use Tox to run tests and linting, e.g.

```shell
pip install tox
```

## Run tests, linting, packaging checks

```shell
tox list                # list all Tox environments
tox run -e ruff         # run code style checks
tox run -e py           # run tests with your default Python
tox run -e package      # verify packaging
tox                     # run all Tox environments
```

Make sure all tests pass before you push your changes to GitHub.
GH Actions will run the tests across all supported Python versions.

## Using arguments (reformat, upload package, help)

You can use additional arguments for the tools called by Tox by
separating them from the Tox arguments by a double-dash `--`, e.g.

```shell
tox run -e ruff -- autograd/core.py --show-source
tox run -e ruff -- autograd/core.py --fix
```

```shell
tox run -e package -- upload
```

```shell
tox run -e py -- --help
```
