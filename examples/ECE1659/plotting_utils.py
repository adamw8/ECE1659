import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from pydrake.all import (
    System
)
from copy import copy

def plot_2d_phase_portrait(f, x1lim=(-1, 1), x2lim=(-1, 1), n=100j, colored=False, **kwargs):
    """
    Code adapted from Russ Tedrake's underactuated course

    Plots the phase portrait for a 2D dynamical system.
    Parameters
    ----------
    f : function
        Callable vectorized function which implements the rhs of the
        state-space dynamics xdot = f(x), with x 2D array.
    x1lim : tuple
        Minimum and maximum values (floats) for the horizontal axis of the
        plot.
    x2lim : tuple
        Minimum and maximum values (floats) for the vertical axis of the
        plot.
    n : complex
        Purely imaginary number with imaginary part equal to the number of knot
        points on each axis of the plot.
    """
    # grid state space, careful here x2 before x1
    X1, X2 = np.mgrid[x1lim[0] : x1lim[1] : n, x2lim[0] : x2lim[1] : n]
    if isinstance(f, System):
        context = f.CreateDefaultContext()
        X1d = copy(X1)
        X2d = copy(X2)
        for i in range(X1.shape[0]):
            for j in range(X1.shape[1]):
                context.SetContinuousState([X1[i, j], X2[i, j]])
                deriv = f.EvalTimeDerivatives(context).CopyToVector()
                X1d[i, j] = deriv[0]
                X2d[i, j] = deriv[1]
    else:
        X1d, X2d = f([X1, X2])

    # color the streamlines according to the magnitude of xdot
    color = np.sqrt(X1d**2 + X2d**2)

    # phase portrait (annoying input format of streamplot)
    if colored:
        strm = plt.streamplot(
            X1.T[0], X2[0], X1d.T, X2d.T, color=color.T, **kwargs
        )

        # colorbar on the right that measures the magnitude of xdot
        plt.gcf().colorbar(strm.lines, label=r"$|\dot\mathbf{x}|$")
    else:
        strm = plt.streamplot(
            X1.T[0], X2[0], X1d.T, X2d.T, color="gray", **kwargs
        )

    # misc plot settings
    plt.title("System Phase Portrait")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.xlim(x1lim)
    plt.ylim(x2lim)
    plt.gca().set_aspect("equal")
    plt.legend(["Vector Field"])

def plot_lyapunov_function(V, x, x1lim=(-1, 1), x2lim=(-1, 1), n=100, levels=None, linewidth=2.5):
    X1 = np.linspace(x1lim[0], x1lim[1], n)
    X2 = np.linspace(x2lim[0], x2lim[1], n)
    Z = np.zeros((n, n))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            Z[j,i] = V.Evaluate({x[0]: x1, x[1]: x2})
    
    cmap = plt.get_cmap('autumn')
    if levels is not None:
        contour = plt.contour(X1, X2, Z, levels=levels, 
                          linestyles='dashed', cmap=cmap, 
                          linewidths=linewidth)
    else:
        contour = plt.contour(X1, X2, Z, 
                          linestyles='dashed', cmap=cmap, 
                          linewidths=linewidth)
    plt.colorbar(contour, label=r"$v(x_1, x_2)$")