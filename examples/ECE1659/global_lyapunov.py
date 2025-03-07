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
    ToLatex
)
import pydrake.symbolic as sym
from pydrake.symbolic import Polynomial
from plotting_utils import *

"""
Code for Global Lyapunov Example

Parts of code are adapted from Russ Tedrake's underactuated course notes.
"""

def find_lyapunov():
    prog = MathematicalProgram()
    x = prog.NewIndeterminates(2, "x")

    # Dynamics of a jet engine: (Derived from Moore-Greitzer)
    f = [-x[1] - 1.5*x[0]**2 - 0.5*x[0]**3, 3*x[0] - x[1]]

    V = prog.NewSosPolynomial(Variables(x), 4)[0].ToExpression()
    prog.AddLinearConstraint(V.Substitute({x[0]: 0, x[1]: 0}) == 0)
    prog.AddLinearConstraint(V.Substitute({x[0]: 1, x[1]: 0}) == 1)
    Vdot = V.Jacobian(x).dot(f)

    prog.AddSosConstraint(-Vdot)

    result = Solve(prog)
    V_certificate = Polynomial(result.GetSolution(V)).RemoveTermsWithSmallCoefficients(1e-6)
    print("$V(x) = "
            + ToLatex(
                V_certificate.ToExpression(), 6,
            )
            + "$")    
    sys = SymbolicVectorSystem(state=x, dynamics=f)
    fig, ax = plt.subplots(figsize=(20, 10))

    lim = 10
    x1lim = (-2*lim,2*lim)
    x2lim = (-lim,lim)
    plot_2d_phase_portrait(sys, x1lim, x2lim, colored=True)
    plt.plot([0], [0], '*', color='red', markersize='10')
    plot_lyapunov_function(V_certificate, x, x1lim, x2lim, levels=np.linspace(0, 3000, 11))
    plt.show()

if __name__ == '__main__':
    find_lyapunov()