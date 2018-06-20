# A demo of gradients through scipy.integrate.odeint,
# estimating the dynamics of a system given a trajectory.

from __future__ import absolute_import
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as npo

import autograd.numpy as np
from autograd import grad
from autograd.scipy.integrate import odeint
from autograd.builtins import tuple
from autograd.misc.optimizers import adam
import autograd.numpy.random as npr


N = 30  # Dataset size
D = 2   # Data dimension
max_T = 1.5

# Two-dimensional damped oscillator
def func(y, t0, A):
    return np.dot(y**3, A)

def nn_predict(inputs, t, params):
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = np.maximum(0, outputs)
    return outputs

def init_nn_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """Build a list of (weights, biases) tuples, one for each layer."""
    return [(rs.randn(insize, outsize) * scale,   # weight matrix
             rs.randn(outsize) * scale)           # bias vector
            for insize, outsize in zip(layer_sizes[:-1], layer_sizes[1:])]

# Define neural ODE model.
def ode_pred(params, y0, t):
    return odeint(nn_predict, y0, t, tuple((params,)), rtol=0.01)

def L1_loss(pred, targets):
    return np.mean(np.abs(pred - targets))

if __name__ == '__main__':

    # Generate data from true dynamics.
    true_y0 = np.array([2., 0.]).T
    t = np.linspace(0., max_T, N)
    true_A = np.array([[-0.1, 2.0], [-2.0, -0.1]])
    true_y = odeint(func, true_y0, t, args=(true_A,))

    def train_loss(params, iter):
        pred = ode_pred(params, true_y0, t)
        return L1_loss(pred, true_y)

    # Set up figure
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj     = fig.add_subplot(131, frameon=False)
    ax_phase    = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)

    # Plots data and learned dynamics.
    def callback(params, iter, g):

        pred = ode_pred(params, true_y0, t)

        print("Iteration {:d} train loss {:.6f}".format(
              iter, L1_loss(pred, true_y)))

        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        ax_traj.plot(t, true_y[:, 0], '-', t, true_y[:, 1], 'g-')
        ax_traj.plot(t, pred[:, 0], '--', t, pred[:, 1], 'b--')
        ax_traj.set_xlim(t.min(), t.max())
        ax_traj.set_ylim(-2, 2)
        ax_traj.xaxis.set_ticklabels([])
        ax_traj.yaxis.set_ticklabels([])
        ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        ax_phase.plot(true_y[:, 0], true_y[:, 1], 'g-')
        ax_phase.plot(pred[:, 0], pred[:, 1], 'b--')
        ax_phase.set_xlim(-2, 2)
        ax_phase.set_ylim(-2, 2)
        ax_phase.xaxis.set_ticklabels([])
        ax_phase.yaxis.set_ticklabels([])

        ax_vecfield.cla()
        ax_vecfield.set_title('Learned Vector Field')
        ax_vecfield.set_xlabel('x')
        ax_vecfield.set_ylabel('y')
        ax_vecfield.xaxis.set_ticklabels([])
        ax_vecfield.yaxis.set_ticklabels([])

        # vector field plot
        y, x = npo.mgrid[-2:2:21j, -2:2:21j]
        dydt = nn_predict(np.stack([x, y], -1).reshape(21 * 21, 2), 0,
            params).reshape(-1, 2)
        mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
        dydt = (dydt / mag)
        dydt = dydt.reshape(21, 21, 2)

        ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
        ax_vecfield.set_xlim(-2, 2)
        ax_vecfield.set_ylim(-2, 2)

        fig.tight_layout()
        plt.draw()
        plt.pause(0.001)


    # Train neural net dynamics to match data.
    init_params = init_nn_params(0.1, layer_sizes=[D, 150, D])
    optimized_params = adam(grad(train_loss), init_params,
                            num_iters=1000, callback=callback)

