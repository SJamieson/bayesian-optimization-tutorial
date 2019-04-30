# Author: Vincent Dubourg <vincent.dubourg@gmail.com>
#         Jake Vanderplas <vanderplas@astro.washington.edu>
# License: BSD style
# Adapted by: Stewart Jamieson

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from matplotlib import pyplot as pl
from matplotlib import animation
from bayes_opt import UtilityFunction

np.random.seed(1)
fps = .5
start = 3.
num_pts = 8
kappa = float(1)
acq_func = UtilityFunction('ucb', kappa=kappa, xi=0)

fixed_pts = [start, 4., 5., 6., 7., 8.]
fixed = True

filename = '{}_demo_k={}'.format('bayesopt' if not fixed else 'fixedpts', kappa)
def f(x):
    """The function to predict."""
    return np.abs(x + np.sin(x) * 4)


# ----------------------------------------------------------------------
# Mesh the input space for evaluations of the real function, the prediction and
# its MSE
x = np.atleast_2d(np.linspace(0, 10, 1000)).T

fig = pl.figure()
pl.plot(x, f(x), 'r:', label=r'Seafloor Depth ($|x\,+\, 4 \sin(x)|$)')
pl.xlabel('$x$')
pl.ylabel('$f(x)$')
pl.ylim(0, 20)

lines = []
points = []
bayesopt, gp = None, None
def draw(frame):
    global lines, points, bayesopt, gp
    num_pts = frame + 1
    if frame == 0:
        points = [[start]]
        # Instanciate a Gaussian Process model
        # bayesopt = BayesianOptimization(f, {'x': (0, 10)}, random_state=100)
        # bayesopt.set_gp_params(kernel=RBF(length_scale=.25))
        gp = GaussianProcessRegressor(kernel=RBF(length_scale=.25))
    for l in lines:
        l.remove()

    if fixed:
        points = fixed_pts[:num_pts]

    X = np.array(points, dtype=np.float64).reshape(-1, 1)
    # bayesopt.register(points[-1], f(points[-1][0]))
    lines = []
    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(X, f(X) / np.max(f(x)))

    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred, sigma = gp.predict(x, return_std=True)
    y_pred, sigma = y_pred * np.max(f(x)), sigma * np.max(f(x))
    sigma = sigma.reshape(y_pred.shape)
    acq = acq_func.utility(x, gp, np.max(f(x))) * np.max(f(x))

    # Plot the function, the prediction and the 95% confidence interval based on
    # the MSE
    lines.extend(pl.plot(X, f(X), 'r.', markersize=10))
    lines.extend(pl.plot(x, y_pred, 'b-', label=u'Prediction'))
    lines.extend(pl.plot(x, acq, color='orange', label=r'UCB ($\kappa = {}$)'.format(kappa)))
    xfill = np.concatenate([x, x[::-1]])
    yfill = np.concatenate([y_pred - 1 * sigma, (y_pred + 1 * sigma)[::-1]])
    lines.extend(pl.fill(xfill, yfill,
                         alpha=.5, fc='b', ec='None', label=r'1$\sigma$ confidence interval'))
    lines.extend(pl.plot([X[-1,0], X[-1,0]], [0, 20], '--m', label='Robot Location'))
    pl.legend(loc='upper left')
    points.append([np.argmax(acq) / 100])

anim = animation.FuncAnimation(fig, draw, frames=num_pts, blit=False)
anim.save(filename + '.gif', writer='imagemagick', fps=fps)
