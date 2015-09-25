from __future__ import absolute_import
from . import norm

# Try block needed in case the user has an
# old version of scipy without multivariate normal.
try:
    from . import multivariate_normal
except AttributeError:
    pass
