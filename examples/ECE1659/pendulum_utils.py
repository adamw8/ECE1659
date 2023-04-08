import numpy as np
import math


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
    b = 0.1
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

def maximize_rho_bilinear(V, xbar, f, iter=3, V_degree=2, lambda_degree=2):
    def optimize_lambda_given_V(V, xbar, f, lambda_degree=2):
        """
        Solves the program:
        
        max rho
        s.t. -Vdot(x) + lambda(x)^T(V(x)-rho) is SOS
        
        using line search.

        Given Variables: V, (Vdot)
        Decision variables: lambda, rho
        """

        # Find Vdot
        Vdot = Jacobian([V], xbar).dot(f)[0][0]
        
        # line search for best rho
        rho = 1
        lo = 0
        hi = -1
        iteration = 0
        best_rho = -1
        best_lambda = None

        while True:
             # Define the SOS program
            prog = MathematicalProgram()
            prog.AddIndeterminates(xbar)

            # add constraints
            lambda_xbar = prog.NewSosPolynomial(
                Variables(xbar), lambda_degree)[0].ToExpression()
            prog.AddSosConstraint(-Vdot + lambda_xbar*(V-rho))
            prog.AddSosConstraint(V)
            prog.AddSosConstraint(lambda_xbar)

            # solve program
            try:
                result = Solve(prog)
            except:
                break

            # line search
            iteration += 1
            if (result.is_success()):
                best_rho = rho
                best_lambda = Polynomial(result.GetSolution(lambda_xbar)).ToExpression()
                lo = rho
                rho = (lo+hi)/2 if hi > 0 else 2*lo
            else:
                hi = rho
                rho = (lo+hi)/2
        
            if (hi > 0 and hi-lo < 0.01) or iteration >= 20:
                break
        
        return best_lambda, best_rho
    
    def optimize_V_given_lambda(lambda_xbar, xbar, f, V_degree=2):
        """
        Solves the program:
        
        min Tr(P)
        s.t. V(x) = m(x)^T*P*m(x)
            (x^T*x)^d*(V(x)-1) + lambda(x)^T*Vdot(x)
             P >= eps*I
        
        Given variables: lambda(x)
        Decision variables: V(x)
        """

        # Create the mathematical program
        for i in range(20):
            rho = 0.9**i
            prog = MathematicalProgram()
            prog.AddIndeterminates(xbar)

            # Create Lyapunov Decision Variable
            (V, P) = prog.NewSosPolynomial(Variables(xbar), V_degree)
            V = V.ToExpression()
            Vdot = Jacobian([V], xbar).dot(f)[0][0]

            # Constraints & Objective
            prog.AddLinearConstraint(V.Substitute({xbar[0]: 0, xbar[1]: 0}) == 0)
            prog.AddSosConstraint(V-1e-9*xbar.dot(xbar))
            prog.AddSosConstraint(-Vdot + lambda_xbar*(V-rho))


            # lambda_degree = Polynomial(lambda_xbar).TotalDegree()
            # Vdot_degree = int(math.ceil(Polynomial(Vdot).TotalDegree() / 2.0) * 2)
            # d = max((lambda_degree + Vdot_degree - V_degree) // 2, 1)
            # prog.AddSosConstraint((V-1)*(xbar.dot(xbar))**d + lambda_xbar * Vdot)
            # prog.AddCost(np.trace(P))

            result = Solve(prog)

            if result.is_success():
                return Polynomial(
                    result.GetSolution(V)
                ).RemoveTermsWithSmallCoefficients(1e-6).ToExpression()

        return None

    Vs = []
    rhos = []
    for i in range(iter):
        print(f"Bilinear iteration {i+1}")
        # Bilinear 1
        lambda_, rho = optimize_lambda_given_V(V, xbar, f, lambda_degree=lambda_degree)
        if lambda_ is None:
            print("Terminated early due to numerical issues in finding lambda.")
            break
        
        Vs.append(V)
        rhos.append(rho)
        
        if i+1 == iter:
            break

        # Bilinear 2
        V = optimize_V_given_lambda(lambda_, xbar, f, V_degree=V_degree)
        if V is None:
            print("Terminated early due to numerical issues in finding V.")
            break

    return Vs, rhos