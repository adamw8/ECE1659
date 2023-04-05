import math

import numpy as np
import matplotlib.pyplot as plt


from pydrake.all import (
    Jacobian,
    MathematicalProgram,
    SymbolicVectorSystem,
    Solve,
    Variables,
    System
)
import pydrake.symbolic as sym
from pydrake.symbolic import Polynomial
from plotting_utils import *

"""
Code for example 1.
"""

def find_lyapunov():
    prog = MathematicalProgram()
    x = prog.NewIndeterminates(2, "x")
    f = [-x[0] - 2 * x[1] ** 2, -x[1] - x[0] * x[1] - 2 * x[1] ** 3]

    V = prog.NewSosPolynomial(Variables(x), 4)[0].ToExpression()
    prog.AddLinearConstraint(V.Substitute({x[0]: 0, x[1]: 0}) == 0)
    prog.AddLinearConstraint(V.Substitute({x[0]: 1, x[1]: 0}) == 1)
    Vdot = V.Jacobian(x).dot(f)

    prog.AddSosConstraint(-Vdot)

    result = Solve(prog)
    V_certificate = Polynomial(result.GetSolution(V)).RemoveTermsWithSmallCoefficients(1e-6)
    print(V_certificate.ToExpression())    
    sys = SymbolicVectorSystem(state=x, dynamics=f)
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_2d_phase_portrait(sys, (-3, 3), (-3, 3))
    plot_lyapunov_function(V_certificate, x, (-3, 3), (-3, 3))
    plt.show()

if __name__ == '__main__':
    find_lyapunov()