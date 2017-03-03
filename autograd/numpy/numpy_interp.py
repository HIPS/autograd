import numpy as np
from autograd import numpy as anp
from autograd import primitive

def interp(x, xp, yp, left=None, right=None):
    """ Differentiable against yp """
    if left is None:
        left = yp[0]
    if right is None: right = yp[-1]

    xp = anp.concatenate([[xp[0]], xp, [xp[-1]]])
    yp = anp.concatenate([anp.array([left]), yp, anp.array([right])])

    m = make_matrix(x, xp)
    y = anp.inner(m, yp)
    return y

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
