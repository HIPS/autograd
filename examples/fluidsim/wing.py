import autograd.numpy as np
from autograd import grad

from scipy.optimize import minimize
from scipy.misc import imread

import matplotlib.pyplot as plt

rows, cols = 55, 55

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
    return np.sum(vy * np.roll(b, -1, axis=0) - vy * np.roll(b, 1, axis=0))

def total_speed(vy):
    return np.sum(vy)

def project(vx, vy, b):
    """Project the velocity field to be approximately mass-conserving,
       using a few iterations of Gauss-Seidel."""
    p = np.zeros(vx.shape)
    h = 1.0 #/vx.shape[0]
    div = -0.5 * h * (np.roll(vx, -1, axis=0) - np.roll(vx, 1, axis=0)
                    + np.roll(vy, -1, axis=1) - np.roll(vy, 1, axis=1))
    div = make_continuous(div, b)

    for k in xrange(20):
        p = (div + np.roll(p, 1, axis=0) + np.roll(p, -1, axis=0)
                 + np.roll(p, 1, axis=1) + np.roll(p, -1, axis=1))/4.0
        p = make_continuous(p, b)

    vx = vx - 0.5*(np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0))/h
    vy = vy - 0.5*(np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1))/h
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


def sigmoid(x):
    return 0.5*(np.tanh(x) + 1.0)   # Output ranges from 0 to 1.

def simulate(vx, vy, num_time_steps, b, ax=None, render=False):
    b = sigmoid(b)

    mask = np.zeros((rows, cols))
    mask[20:40, 20:40] = 1.0
    c = 1 - b
    c = c * mask
    b = 1 - c

    red_smoke = np.zeros((rows, cols))
    red_smoke[rows/4:rows/2] = 1
    blue_smoke = np.zeros((rows, cols))
    blue_smoke[rows/2:3*rows/4] = 1

    print "Running simulation..."
    for t in xrange(num_time_steps):
        if ax: plot_matrix(ax, red_smoke, blue_smoke, 1 - b, num_time_steps, render)
        vx_updated = advect(vx, vx, vy)
        vy_updated = advect(vy, vx, vy)
        vx, vy = project(vx_updated, vy_updated, b)
        red_smoke = advect(red_smoke, vx, vy)
        red_smoke = block(red_smoke, b)
        blue_smoke = advect(blue_smoke, vx, vy)
        blue_smoke = block(blue_smoke, b)
    return vx

def plot_matrix(ax, r, b, g, t, render=False):
    plt.cla()
    ax.imshow(np.concatenate((r[...,np.newaxis], g[...,np.newaxis], b[...,np.newaxis]), axis=2))
    ax.set_xticks([])
    ax.set_yticks([])
    plt.draw()
    if render:
        plt.savefig('step{0:03d}.png'.format(t), bbox_inches='tight')
    plt.pause(0.001)


if __name__ == '__main__':

    simulation_timesteps = 40

    print "Loading initial and target states..."
    init_vx = np.zeros((rows, cols))
    init_vy = np.ones((rows, cols))

    init_b = np.ones((rows, cols))
    init_b[25:35, 25:35] = 0.0
    init_b = init_b.ravel()

    def objective(params):
        cur_b = np.reshape(params, (rows, cols))
        final_vx = simulate(init_vx, init_vy, simulation_timesteps, cur_b)
        return total_speed(final_vx) #updraft(final_vx, cur_b)

    # Specify gradient of objective function using autograd.
    objective_with_grad = grad(objective, return_function_value=True)

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, frameon=False)

    def callback(weights):
        cur_b = np.reshape(weights, (rows, cols))
        simulate(init_vx, init_vy, simulation_timesteps, cur_b, ax)

    print "Rendering initial flow..."
    callback(init_b)

    print "Optimizing initial conditions..."
    result = minimize(objective_with_grad, init_b, jac=True, method='CG',
                      options={'maxiter':25, 'disp':True}, callback=callback)

    print "Rendering optimized flow..."
    cur_b = np.reshape(result.x, (rows, cols))
    simulate(init_vx, init_vy, simulation_timesteps, cur_b, ax, render=False)

    #print "Converting frames to an animated GIF..."
    #os.system("convert -delay 5 -loop 0 step*.png"
    #          " -delay 250 step099.png surprise.gif")  # Using imagemagick.
    #os.system("rm step*.png")
