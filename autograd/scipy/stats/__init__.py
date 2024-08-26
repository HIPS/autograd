from . import beta, chi2, gamma, norm, poisson, t

# Try block needed in case the user has an
# old version of scipy without multivariate normal.
try:
    from . import multivariate_normal
except AttributeError:
    pass

try:
    from . import dirichlet
except AttributeError:
    pass
