from . import numpy_wrapper as anp

def _interp_vjp(x, xp, yp, left, right, period, g):
    from autograd import vector_jacobian_product
    func = vector_jacobian_product(_interp, argnum=2)
    return func(x, xp, yp, left, right, period, g)

def _interp(x, xp, yp, left=None, right=None, period=None):
    """ A partial rewrite of interp that is differentiable against yp """
    if period is not None:
        xp = anp.concatenate([[xp[-1] - period], xp, [xp[0] + period]])
        yp = anp.concatenate([anp.array([yp[-1]]), yp, anp.array([yp[0]])])
        return _interp(x % period, xp, yp, left, right, None)

    if left is None: left = yp[0]
    if right is None: right = yp[-1]

    xp = anp.concatenate([[xp[0]], xp, [xp[-1]]])

    yp = anp.concatenate([anp.array([left]), yp, anp.array([right])])
    m = make_matrix(x, xp)
    y = anp.inner(m, yp)
    return y

anp.interp.defvjp(lambda g, ans, vs, gvs, x, xp, yp, left=None, right=None, period=None:
    _interp_vjp(x, xp, yp, left, right, period, g), argnum=2)


# The following are internal functions

import numpy as np

def W(r, D):
    """ Convolution kernel for linear interpolation.
        D is the differences of xp.
    """
    mask = D == 0
    D[mask] = 1.0
    Wleft = 1.0 + r[1:] / D
    Wright = 1.0 - r[:-1] / D
    # edges
    Wleft = np.where(mask, 0, Wleft)
    Wright = np.where(mask, 0, Wright)
    Wleft = np.concatenate([[0], Wleft])
    Wright = np.concatenate([Wright, [0]])
    W = np.where(r < 0, Wleft, Wright)
    W = np.where(r == 0, 1.0, W)
    W = np.where(W < 0, 0, W)
    return W

def make_matrix(x, xp):
    D = np.diff(xp)
    w = []
    v0 = np.zeros(len(xp))
    v0[0] = 1.0
    v1 = np.zeros(len(xp))
    v1[-1] = 1.0
    for xi in x:
        # left, use left
        if xi < xp[0]: v = v0
        # right , use right
        elif xi > xp[-1]: v = v1
        else:
            v = W(xi - xp, D)
            v[0] = 0
            v[-1] = 0
        w.append(v)
    return np.array(w)
