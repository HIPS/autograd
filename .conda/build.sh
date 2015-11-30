#!/bin/bash

export USE_CYTHON=True

$PYTHON -c "from version import __version__; print(__version__)" > __conda_version__.txt
$PYTHON setup.py install

# Add more build steps here, if they are necessary.

# See
# http://docs.continuum.io/conda/build.html
# for a list of environment variables that are set during the build process.
