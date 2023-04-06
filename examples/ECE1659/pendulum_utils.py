import numpy as np


from pydrake.all import (
    Jacobian,
    MathematicalProgram,
    Solve,
    Variables,
    DiagramBuilder,
    LinearQuadraticRegulator,
    Linearize
)
from pydrake.examples import PendulumPlant
import pydrake.symbolic as sym
from pydrake.symbolic import Polynomial

def verify_rho(rho):
    # Setup system
    builder = DiagramBuilder()
    pendulum = builder.AddSystem(PendulumPlant())
    context = pendulum.CreateDefaultContext()
    pendulum.get_input_port(0).FixValue(context, [0])
    context.SetContinuousState([np.pi, 0])

    # linearize the system at [pi, 0]
    linearized_pendulum = Linearize(pendulum, context)

    # LQR
    Q = np.diag((5., 1.))
    R = [1.]
    (K, P) = LinearQuadraticRegulator(linearized_pendulum.A(), linearized_pendulum.B(), Q, R)

    # Pendulum Parameters
    m = 1
    l = 0.5
    b = 0.1 # no damping
    g = 9.81
    
    # Define the SOS program
    prog = MathematicalProgram()
    xbar = prog.NewIndeterminates(2, "xbar")

    # Closed loop dyanmics using the Taylor approximation of sin at xbar = x-[pi, 0]^T
    xbar1 = xbar[0]
    xbar2 = xbar[1]
    T_sin = -(xbar1 - xbar1**3/6)
    f = [xbar2, (-K.dot(xbar)-b*xbar2 - m*g*l*T_sin)/(m*l*l)]

    # Use cost-to-go as the Lyapunov candidate
    V = xbar.dot(P.dot(xbar))
    Vdot = Jacobian([V], xbar).dot(f)[0][0]

    # add constraints
    lambda_xbar = prog.NewSosPolynomial(Variables(xbar), 2)[0].ToExpression()
    prog.AddSosConstraint(-Vdot + lambda_xbar*(V-rho))
    prog.AddSosConstraint(V)
    prog.AddSosConstraint(lambda_xbar)

    result = Solve(prog)
    return pendulum, V, xbar, result.is_success()

def maximize_rho():
    # line search for largest rho
    rho = 1
    best_rho = -1
    lo = 1
    hi = -1
    iteration = 0

    while True:
        (pendulum, V, xbar, is_certified) = verify_rho(rho)
        iteration += 1
        if (is_certified):
            best_rho = rho
            lo = rho
            rho = (lo+hi)/2 if hi > 0 else 2*lo
            print(f"V(x) < {rho} is a region of attraction.")
        else:
            hi = rho
            rho = (lo+hi)/2
            print(f"V(x) < {rho} is not a certifiable region of attraction.")

        if (hi > 0 and hi-lo < 1) or iteration >= 20:
            break
    
    (pendulum, V, xbar, is_certified) = verify_rho(best_rho)
    return (pendulum, V, xbar, best_rho)