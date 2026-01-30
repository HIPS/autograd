# Autograd examples

## Usage instructions

Some of the examples require additional dependencies beyond Autograd and its
core dependencies. These are set up under the `examples` dependency group. To
install them, navigate to the root directory of where you cloned Autograd and
run
```sh
pip install --group examples
```
from the command line. Note that dependency groups are a recent feature so you
may need to upgrade `pip` with
```sh
pip install --upgrade pip
```

Having installed the additional dependencies, you may navigate to the `examples`
subdirectory and run any of the Python scripts. For example:
```sh
python3 tanh.py
```
Some of the examples print to the terminal and others open pop-up windows for
plots.
