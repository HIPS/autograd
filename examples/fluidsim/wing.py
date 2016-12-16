from __future__ import absolute_import
from __future__ import print_function
import autograd.numpy as np
from autograd import value_and_grad

from scipy.optimize import minimize

import matplotlib.pyplot as plt
import os
from builtins import range

rows, cols = 40, 60

# Fluid simulation code based on
# "Real-Time Fluid Dynamics for Games" by Jos Stam
# http://www.intpowertechcorp.com/GDC03.pdf

def occlude(f, occlusion):
    return f * (1 - occlusion)

def project(vx, vy, occlusion):
    """Project the velocity field to be approximately mass-conserving,
       using a few iterations of Gauss-Seidel."""
    p = np.zeros(vx.shape)
    div = -0.5 * (np.roll(vx, -1, axis=1) - np.roll(vx, 1, axis=1)
                + np.roll(vy, -1, axis=0) - np.roll(vy, 1, axis=0))
    div = make_continuous(div, occlusion)

    for k in range(50):
        p = (div + np.roll(p, 1, axis=1) + np.roll(p, -1, axis=1)
                 + np.roll(p, 1, axis=0) + np.roll(p, -1, axis=0))/4.0
        p = make_continuous(p, occlusion)

    vx = vx - 0.5*(np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1))
    vy = vy - 0.5*(np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0))

    vx = occlude(vx, occlusion)
    vy = occlude(vy, occlusion)
    return vx, vy

def advect(f, vx, vy):
    """Move field f according to x and y velocities (u and v)
       using an implicit Euler integrator."""
    rows, cols = f.shape
    cell_xs, cell_ys = np.meshgrid(np.arange(cols), np.arange(rows))
    center_xs = (cell_xs - vx).ravel()
    center_ys = (cell_ys - vy).ravel()

    # Compute indices of source cells.
    left_ix = np.floor(center_ys).astype(int)
    top_ix  = np.floor(center_xs).astype(int)
    rw = center_ys - left_ix              # Relative weight of right-hand cells.
    bw = center_xs - top_ix               # Relative weight of bottom cells.
    left_ix  = np.mod(left_ix,     rows)  # Wrap around edges of simulation.
    right_ix = np.mod(left_ix + 1, rows)
    top_ix   = np.mod(top_ix,      cols)
    bot_ix   = np.mod(top_ix  + 1, cols)

    # A linearly-weighted sum of the 4 surrounding cells.
    flat_f = (1 - rw) * ((1 - bw)*f[left_ix,  top_ix] + bw*f[left_ix,  bot_ix]) \
                 + rw * ((1 - bw)*f[right_ix, top_ix] + bw*f[right_ix, bot_ix])
    return np.reshape(flat_f, (rows, cols))

def make_continuous(f, occlusion):
    non_occluded = 1 - occlusion
    num = np.roll(f,  1, axis=0) * np.roll(non_occluded,  1, axis=0)\
        + np.roll(f, -1, axis=0) * np.roll(non_occluded, -1, axis=0)\
        + np.roll(f,  1, axis=1) * np.roll(non_occluded,  1, axis=1)\
        + np.roll(f, -1, axis=1) * np.roll(non_occluded, -1, axis=1)
    den = np.roll(non_occluded,  1, axis=0)\
        + np.roll(non_occluded, -1, axis=0)\
        + np.roll(non_occluded,  1, axis=1)\
        + np.roll(non_occluded, -1, axis=1)
    return f * non_occluded + (1 - non_occluded) * num / ( den + 0.001)

def sigmoid(x):
    return 0.5*(np.tanh(x) + 1.0)   # Output ranges from 0 to 1.

def simulate(vx, vy, num_time_steps, occlusion, ax=None, render=False):
    occlusion = sigmoid(occlusion)

    # Disallow occlusion outside a certain area.
    mask = np.zeros((rows, cols))
    mask[10:30, 10:30] = 1.0
    occlusion = occlusion * mask

    # Initialize smoke bands.
    red_smoke = np.zeros((rows, cols))
    red_smoke[rows//4:rows//2] = 1
    blue_smoke = np.zeros((rows, cols))
    blue_smoke[rows//2:3*rows//4] = 1

    print("Running simulation...")
    vx, vy = project(vx, vy, occlusion)
    for t in range(num_time_steps):
        plot_matrix(ax, red_smoke, occlusion, blue_smoke, t, render)
        vx_updated = advect(vx, vx, vy)
        vy_updated = advect(vy, vx, vy)
        vx, vy = project(vx_updated, vy_updated, occlusion)
        red_smoke = advect(red_smoke, vx, vy)
        red_smoke = occlude(red_smoke, occlusion)
        blue_smoke = advect(blue_smoke, vx, vy)
        blue_smoke = occlude(blue_smoke, occlusion)
    plot_matrix(ax, red_smoke, occlusion, blue_smoke, num_time_steps, render)
    return vx, vy

def plot_matrix(ax, r, g, b, t, render=False):
    if ax:
        plt.cla()
        ax.imshow(np.concatenate((r[...,np.newaxis], g[...,np.newaxis], b[...,np.newaxis]), axis=2))
        ax.set_xticks([])
        ax.set_yticks([])
        plt.draw()
        if render:
            plt.savefig('step{0:03d}.png'.format(t), bbox_inches='tight')
        plt.pause(0.001)


if __name__ == '__main__':

    simulation_timesteps = 20

    print("Loading initial and target states...")
    init_vx = np.ones((rows, cols))
    init_vy = np.zeros((rows, cols))

    # Initialize the occlusion to be a block.
    init_occlusion = -np.ones((rows, cols))
    init_occlusion[15:25, 15:25] = 0.0
    init_occlusion = init_occlusion.ravel()

    def drag(vx): return np.mean(init_vx - vx)
    def lift(vy): return np.mean(vy - init_vy)

    def objective(params):
        cur_occlusion = np.reshape(params, (rows, cols))
        final_vx, final_vy = simulate(init_vx, init_vy, simulation_timesteps, cur_occlusion)
        return -lift(final_vy) / drag(final_vx)

    # Specify gradient of objective function using autograd.
    objective_with_grad = value_and_grad(objective)

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, frameon=False)

    def callback(weights):
        cur_occlusion = np.reshape(weights, (rows, cols))
        simulate(init_vx, init_vy, simulation_timesteps, cur_occlusion, ax)

    print("Rendering initial flow...")
    callback(init_occlusion)

    print("Optimizing initial conditions...")
    result = minimize(objective_with_grad, init_occlusion, jac=True, method='CG',
                      options={'maxiter':50, 'disp':True}, callback=callback)

    print("Rendering optimized flow...")
    final_occlusion = np.reshape(result.x, (rows, cols))
    simulate(init_vx, init_vy, simulation_timesteps, final_occlusion, ax, render=True)

    print("Converting frames to an animated GIF...")   # Using imagemagick.
    os.system("convert -delay 5 -loop 0 step*.png "
              "-delay 250 step{0:03d}.png wing.gif".format(simulation_timesteps))
    os.system("rm step*.png")
