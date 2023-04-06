import math

import numpy as np
import matplotlib.pyplot as plt


from pydrake.all import (
    Jacobian,
    MathematicalProgram,
    SymbolicVectorSystem,
    Solve,
    Variables,
    System,
    ToLatex,
    DiagramBuilder,
    LinearQuadraticRegulator,
    Linearize
)
from pydrake.examples import PendulumPlant
import pydrake.symbolic as sym
from pydrake.symbolic import Polynomial
from plotting_utils import *
from pendulum_utils import *
"""
Code for Region of Attraction Example
"""

def pendulum_ROA():
    rho = 1
    (pendulum, V, xbar, is_certified) = verify_rho(rho)
    # (pendulum, V, xbar, rho) = maximize_rho()

    # visualize the dynamics
    # sys = SymbolicVectorSystem(state=x, dynamics=f)
    fig, ax = plt.subplots(figsize=(10, 20))
    xlim = (-2*np.pi,2*np.pi)
    ylim = (-5,5)

    plot_2d_phase_portrait(pendulum, xlim, ylim, colored=True)
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\dot{\theta}$")
    # plot markers at the stable and unstable equillibrium
    plt.plot([-2*np.pi, 0, 2*np.pi], [0, 0, 0], 'x', color='red', markersize='10')
    plt.plot([-np.pi, np.pi], [0, 0], '*', color='red', markersize='10')

    # plot the certified ROA
    n = 100
    X1 = np.linspace(xlim[0], xlim[1], n)
    X2 = np.linspace(ylim[0], ylim[1], n)
    Z = np.zeros((n,n))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            Z[j,i] = V.Evaluate({xbar[0]: x1-np.pi, xbar[1]: x2})
    plt.contour(X1, X2, Z, levels=[rho], 
                        linestyles='dashed', cmap=plt.get_cmap('autumn'), 
                        linewidths=2)

    plt.show()
    

if __name__ == '__main__':
    pendulum_ROA()