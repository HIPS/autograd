# distutils: extra_compile_args = -Ofast -w
# cython: boundscheck=False, nonecheck=False, wraparound=False, cdivision=True

# based on code by James Hensman and Alan Saul, 2015, used with permission
# https://github.com/SheffieldML/GPy/blob/4be3bd33e0110b87c2fd9fd39abb331858a6c7df/GPy/util/choleskies_cython.pyx

import numpy as np
cimport numpy as np
from scipy.linalg.cython_blas cimport dsymv, ddot
from cython cimport floating

cdef inline void _cholesky_grad(double[::1,:] L, double[::1,:] dL):
    cdef int i, k, n, N = L.shape[0], inc = 1
    cdef double neg1 = -1, one = 1

    dL[N-1,N-1] /= 2. * L[N-1,N-1]
    for k in range(N-2, -1, -1):
        n = N-k-1
        dsymv('L', &n, &neg1, &dL[k+1,k+1], &N, &L[k+1,k], &inc, &one, &dL[k+1,k], &inc)
        for i in range(N-k-1):
            dL[k+1+i,k] -= dL[k+1+i,k+1+i] * L[k+1+i,k]
            dL[k+1+i,k] /= L[k,k]
        dL[k,k] -= ddot(&n, &dL[k+1,k], &inc, &L[k+1,k], &inc)
        dL[k,k] /= 2. * L[k,k]

def cholesky_grad(floating[:,:] L, floating[:,:] dL):
    cdef double[::1,:] _dL = np.require(np.tril(dL), np.double, 'F')
    cdef double[::1,:] _L = np.require(L, np.double, 'F')
    _cholesky_grad(_L, _dL)
    return (np.asarray(_dL) + np.asarray(_dL).T)/2.
