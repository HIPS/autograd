import autograd.numpy as np
from autograd import grad

from scipy.optimize import minimize
from scipy.misc import imread

import matplotlib.pyplot as plt
import os

rows, cols = 110, 110

# Fluid simulation code based on
# "Real-Time Fluid Dynamics for Games" by Jos Stam
# http://www.intpowertechcorp.com/GDC03.pdf



def make_continuous(f, b):
    num = np.roll(f,  1, axis=0) * np.roll(b,  1, axis=0)\
        + np.roll(f, -1, axis=0) * np.roll(b, -1, axis=0)\
        + np.roll(f,  1, axis=1) * np.roll(b,  1, axis=1)\
        + np.roll(f, -1, axis=1) * np.roll(b, -1, axis=1)
    den = np.roll(b,  1, axis=0)\
        + np.roll(b, -1, axis=0)\
        + np.roll(b,  1, axis=1)\
        + np.roll(b, -1, axis=1)
    return f * b + (1 - b) * num / ( den + 0.01)

def reflect(f, b, axis):
    c = 1 - b
    num = np.roll(f,  1, axis) * np.roll(c,  1, axis)\
        + np.roll(f, -1, axis) * np.roll(c, -1, axis)
    den = np.roll(c,  1, axis)\
        + np.roll(c, -1, axis)
    return f * b - c  * num / ( den + 0.01)

def block(f, b):
    return f * b

def updraft(vy, b):
    return vy * np.roll(b, -1, axis=0) - vy * np.roll(b, 1, axis=0)

def project(vx, vy, b):
    """Project the velocity field to be approximately mass-conserving,
       using a few iterations of Gauss-Seidel."""
    p = np.zeros(vx.shape)
    h = 1.0 #/vx.shape[0]
    div = -0.5 * h * (np.roll(vx, -1, axis=0) - np.roll(vx, 1, axis=0)
                    + np.roll(vy, -1, axis=1) - np.roll(vy, 1, axis=1))
    div = make_continuous(div, b)

    for k in xrange(100):
        p = (div + np.roll(p, 1, axis=0) + np.roll(p, -1, axis=0)
                 + np.roll(p, 1, axis=1) + np.roll(p, -1, axis=1))/4.0
        p = make_continuous(p, b)

    vx -= 0.5*(np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0))/h
    vy -= 0.5*(np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1))/h
    #vx = reflect(vx, b, 0)
    #vy = reflect(vy, b, 1)
    vx = block(vx, b)
    vy = block(vy, b)
    return vx, vy

def advect(f, vx, vy):
    """Move field f according to x and y velocities (u and v)
       using an implicit Euler integrator."""
    rows, cols = f.shape
    cell_ys, cell_xs = np.meshgrid(np.arange(rows), np.arange(cols))
    center_xs = (cell_xs - vx).ravel()
    center_ys = (cell_ys - vy).ravel()

    # Compute indices of source cells.
    left_ix = np.floor(center_xs).astype(np.int)
    top_ix  = np.floor(center_ys).astype(np.int)
    rw = center_xs - left_ix              # Relative weight of right-hand cells.
    bw = center_ys - top_ix               # Relative weight of bottom cells.
    left_ix  = np.mod(left_ix,     rows)  # Wrap around edges of simulation.
    right_ix = np.mod(left_ix + 1, rows)
    top_ix   = np.mod(top_ix,      cols)
    bot_ix   = np.mod(top_ix  + 1, cols)

    # A linearly-weighted sum of the 4 surrounding cells.
    flat_f = (1 - rw) * ((1 - bw)*f[left_ix,  top_ix] + bw*f[left_ix,  bot_ix]) \
                 + rw * ((1 - bw)*f[right_ix, top_ix] + bw*f[right_ix, bot_ix])
    return np.reshape(flat_f, (rows, cols))

def simulate(vx, vy, smoke, num_time_steps, b, ax=None, render=False):
    print "Running simulation..."
    for t in xrange(num_time_steps):
        if ax: plot_matrix(ax, smoke, t, render)
        vx_updated = advect(vx, vx, vy)
        vy_updated = advect(vy, vx, vy)
        vx, vy = project(vx_updated, vy_updated, b)
        smoke = advect(smoke, vx, vy)
        smoke = block(smoke, b)
    if ax: plot_matrix(ax, smoke, num_time_steps, render)
    return smoke

def plot_matrix(ax, mat, t, render=False):
    plt.cla()
    ax.matshow(mat)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.draw()
    if render:
        plt.savefig('step{0:03d}.png'.format(t), bbox_inches='tight')
    plt.pause(0.001)


if __name__ == '__main__':

    simulation_timesteps = 20

    print "Loading initial and target states..."
    init_smoke = imread('skull.png')[::2,::2].view(np.ndarray)
    init_dx_and_dy = np.zeros((2, rows, cols)).ravel()
    init_dx_and_dy[(rows*cols):] = 1.0

    b = np.ones((rows, cols))
    b[50:80, 50:80] = 0.0

    def convert_param_vector_to_matrices(params):
        vx = np.reshape(params[:(rows*cols)], (rows, cols))
        vy = np.reshape(params[(rows*cols):], (rows, cols))
        return vx, vy

    #def objective(params):
    #    init_vx, init_vy = convert_param_vector_to_matrices(params)
    #    final_smoke = simulate(init_vx, init_vy, init_smoke, simulation_timesteps)
    #    return distance_from_target_image(final_smoke)

    # Specify gradient of objective function using autograd.
    #objective_with_grad = grad(objective, return_function_value=True)

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, frameon=False)

    #def callback(weights):
    #    init_vx = np.reshape(weights[0:(rows*cols)], (rows, cols))
    #    init_vy = np.reshape(weights[(rows*cols):], (rows, cols))
    #    simulate(init_vx, init_vy, init_smoke, simulation_timesteps, ax)

    #print "Optimizing initial conditions..."
    #result = minimize(objective_with_grad, init_dx_and_dy, jac=True, method='CG',
    #                  options={'maxiter':25, 'disp':True}, callback=callback)

    #print "Rendering optimized flow..."
    init_vx, init_vy = convert_param_vector_to_matrices(init_dx_and_dy)
    simulate(init_vx, init_vy, init_smoke, simulation_timesteps, b, ax, render=False)

    #print "Converting frames to an animated GIF..."
    #os.system("convert -delay 5 -loop 0 step*.png"
    #          " -delay 250 step099.png surprise.gif")  # Using imagemagick.
    #os.system("rm step*.png")
