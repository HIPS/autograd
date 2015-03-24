import autograd.numpy as np
from autograd import grad

from scipy.optimize import fmin_cg

import matplotlib.pyplot as plt

# Based on http://www.intpowertechcorp.com/GDC03.pdf

rows = 100
cols = 100
dt = 3.1
num_timesteps = 25
num_solver_iters = 5

def plot_matrix(ax, mat):
    plt.cla()
    ax.matshow(mat)
    plt.draw()
    plt.pause(0.001)

def project(vx, vy):
    """Project the velocity field to be mass-conserving,
       again using a few iterations of Gauss-Seidel."""
    p = np.zeros((rows, cols))
    h = 1.0/(max(rows,cols));
    div = -0.5 * h * (np.roll(vx, -1, axis=0) - np.roll(vx, 1, axis=0)
                    + np.roll(vy, -1, axis=1) - np.roll(vy, 1, axis=1))

    for k in xrange(num_solver_iters):
        p = (div + np.roll(p, 1, axis=0) + np.roll(p, -1, axis=0)
                 + np.roll(p, 1, axis=1) + np.roll(p, -1, axis=1))/4.0

    vx -= 0.5*(np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0))/h;
    vy -= 0.5*(np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1))/h;
    return vx, vy

def advect(f, vx, vy):
    """Move field f according to x and y velocities (u and v)
       using an implicit Euler integrator."""
    cell_ys, cell_xs = np.meshgrid(np.arange(rows), np.arange(cols))
    center_xs = (cell_xs - dt * vx).ravel()
    center_ys = (cell_ys - dt * vy).ravel()

    # Compute indices of source cells.
    left_ix = np.floor(center_xs).astype(np.int)
    top_ix  = np.floor(center_ys).astype(np.int)
    rw = center_xs - left_ix              # Relative weight of right-hand cells.
    bw = center_ys - top_ix               # Relative weight of bottom cells.
    left_ix  = np.mod(left_ix,     rows)  # Wrap around edges of simulation.
    right_ix = np.mod(left_ix + 1, rows)
    top_ix   = np.mod(top_ix,      cols)
    bot_ix   = np.mod(top_ix + 1,  cols)

    # A linearly-weighted sum of the 4 surrounding cells.
    flat_f = (1 - rw) * ((1 - bw)*f[left_ix,  top_ix] + bw*f[left_ix,  bot_ix]) \
                 + rw * ((1 - bw)*f[right_ix, top_ix] + bw*f[right_ix, bot_ix])
    return np.reshape(flat_f, (rows, cols))

def simulate(vx, vy, smoke, num_time_steps, ax=None):
    """Simulate a fluid for a number of time steps."""
    for t in xrange(num_time_steps):
        vx_updated = advect(vx, vx, vy)
        vy_updated = advect(vy, vx, vy)
        vx, vy = project(vx_updated, vy_updated)
        smoke = advect(smoke, vx, vy)
        if ax: plot_matrix(ax, smoke)
    return smoke

def target_match(smoke):
    """Compute distance from target image."""
    target = np.zeros((rows, cols))
    target[30:60, 30:60] = 1.0
    return np.sum(target * smoke)

if __name__ == '__main__':

    init_dx_and_dy = np.zeros((2, rows, cols)).ravel()
    #init_dx = np.random.randn(rows, cols) * 3.
    #init_dy = np.random.randn(rows, cols) * 3.
    init_smoke = np.zeros((rows, cols))
    init_smoke[10:20:, :] = 1.0
    init_smoke[50:60:, :] = 1.0

    fig = plt.figure(figsize=(12,10))
    ax = fig.add_axes([0., 0., 1., 1.], frameon=False)

    #simulate(init_dx, init_dy, init_smoke, num_timesteps, ax)
    def objective(init_vx_and_vy):
        init_vx = np.reshape(init_vx_and_vy[0:(rows*cols)], (rows, cols))
        init_vy = np.reshape(init_vx_and_vy[(rows*cols):], (rows, cols))
        #init_vx = init_vx_and_vy[0, :, :]
        #init_vy = init_vx_and_vy[1, :, :]

        final_smoke = simulate(init_vx, init_vy, init_smoke, num_timesteps)
        return target_match(final_smoke)

    grad_obj = grad(objective)
    #g = grad_obj(init_dx_and_dy, init_smoke, num_timesteps)
    #plot_matrix(ax, g[0])

    def callback(weights):
        init_vx = np.reshape(weights[0:(rows*cols)], (rows, cols))
        init_vy = np.reshape(weights[(rows*cols):], (rows, cols))
        simulate(init_vx, init_vy, init_smoke, num_timesteps, ax)

    weights = fmin_cg(objective, init_dx_and_dy, fprime=grad_obj,
                      maxiter=100, callback=callback)

    init_vx = np.reshape(weights[0:(rows*cols)], (rows, cols))
    init_vy = np.reshape(weights[(rows*cols):], (rows, cols))
    simulate(init_vx, init_vy, init_smoke, num_timesteps, ax)

