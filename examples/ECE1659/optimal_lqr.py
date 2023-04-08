import math

import numpy as np
import matplotlib.pyplot as plt
import argparse

from plotting_utils import *
from pendulum_utils import *

from pydrake.all import (
    MathematicalProgram,
    SymbolicVectorSystem,
)

def compute_ROA_volume(V, x, rho):
    V_normalized = Polynomial(V / rho)

    prog = MathematicalProgram()
    prog.AddIndeterminates(x)

    (V_prog, P) = prog.NewSosPolynomial(Variables(x), 2)
    prog.AddEqualityConstraintBetweenPolynomials(V_normalized, V_prog)
    result = Solve(prog)
    return np.trace(np.linalg.inv(result.GetSolution(P)[0:2,0:2]))


def optimal_lqr(RL_iter=10, bilinear_iter = 3):
    # Setup linearized system
    A = np.array([[-1, 0], [0, 1]])
    B = np.ones((2,1))
    Q = np.diag((1., 1.))
    R = [1]

    alpha = 0.5
    baseline_volume = -1
    best_Q = Q
    best_R = R
    best_volume = -1

    # Search for LQR controller with largest ROA using weight perturbation
    for iter in range(RL_iter):
        """ Compute the new baseline volume and plot it """
        (K, P) = LinearQuadraticRegulator(A, B, Q, R)

        # Define the SOS program
        prog = MathematicalProgram()
        x = prog.NewIndeterminates(2, "x")

        # Closed loop dyanmics
        u = -K.dot(x)
        f = [-x[0] - 2*x[1]**2 + u,
            x[1] + x[0]*x[1] + 2*x[1]**3 + u]
        
        # Use cost-to-go as the Lyapunov candidate
        V = x.dot(P.dot(x))

        # Bilinear seaerch for ROA
        Vs, rhos = maximize_rho_bilinear(
            V, x, f, bilinear_iter, V_degree=2, lambda_degree=6)
        if len(Vs) == 0:
            print("Failed to certify any ROA with current Q")
            return
        
        baseline_volume = compute_ROA_volume(Vs[-1], x, rhos[-1])
        if baseline_volume > best_volume:
            best_volume = baseline_volume
            best_Q = Q
            best_R = R
        
        print(Q)
        print(R)
        print(baseline_volume)

        # visualize the dynamics
        sys = SymbolicVectorSystem(state=x, dynamics=f)
        fig, ax = plt.subplots(figsize=(10, 20))
        xlim = (-4,4)
        ylim = (-4,4)

        plot_2d_phase_portrait(sys, xlim, ylim, colored=True)
        plt.plot([0], [0], '*', color='red', markersize='10')

        # plot the certified ROA       
        n = 100
        X1 = np.linspace(xlim[0], xlim[1], n)
        X2 = np.linspace(ylim[0], ylim[1], n)
        Z = np.zeros((n,n))

        V = Vs[-1]
        rho = rhos[-1]
        for i, x1 in enumerate(X1):
            for j, x2 in enumerate(X2):
                Z[j,i] = V.Evaluate({x[0]: x1, x[1]: x2})
        plt.contour(X1, X2, Z, levels=[rho], 
                            linestyles='dashed', cmap=plt.get_cmap('autumn'), 
                            linewidths=2)

        plt.show()

        """ Perform a random perturbation """
        # Search for a feasible perturbation on Q
        while True:
            dQ = np.array([[np.random.randn(), 0],
                           [ 0, np.random.randn()]])
            new_Q = Q + dQ
            # check that new_Q is PSD
            if new_Q[0][0] >= 0 and new_Q[1][1] >= 0:
                break
        
        # Search for a feasible perturbation on R
        while True:
            dR = np.random.randn(1)
            print(type(R), type(dR))
            new_R = R + dR
            # check that new_R is PD
            if new_R > 0:
                break

        # Evalute volume of new_Q
        (K, P) = LinearQuadraticRegulator(A, B, new_Q, new_R)

        # Define the SOS program
        prog = MathematicalProgram()
        x = prog.NewIndeterminates(2, "x")

        # Closed loop dyanmics
        u = -K.dot(x)
        f = [-x[0] - 2*x[1]**2 + u,
            x[1] + x[0]*x[1] + 2*x[1]**3 + u]
        
        # Use cost-to-go as the Lyapunov candidate
        V = x.dot(P.dot(x))

        # Bilinear seaerch for ROA
        Vs, rhos = maximize_rho_bilinear(
            V, x, f, bilinear_iter, V_degree=2, lambda_degree=6)
        
        if len(Vs) == 0:
            print("Failed to certify any ROA with new_Q")
            return
        
        # Update Q
        new_volume = compute_ROA_volume(Vs[-1], x, rhos[-1])
        alpha_temp = alpha
        while True:
            Q_perturbed = Q + alpha_temp*(new_volume-baseline_volume)*dQ
            R_perturbed = R + alpha_temp*(new_volume-baseline_volume)*dR
            if Q_perturbed[0][0] >= 0 and Q_perturbed[1][1] >= 0 and R_perturbed[0] > 0:
                Q = Q_perturbed
                R = R_perturbed
                break
            else:
                alpha_temp /= 2.

                

    """ RL Search Complete. Plot best V, rho, K """
    # Define the SOS program
    prog = MathematicalProgram()
    x = prog.NewIndeterminates(2, "x")

    # Closed loop dyanmics
    (best_K, best_P) = LinearQuadraticRegulator(A, B, best_Q, best_R)
    u = -best_K.dot(x)
    f = [-x[0] - 2*x[1]**2 + u,
        x[1] + x[0]*x[1] + 2*x[1]**3 + u]
    
    # Use cost-to-go as the Lyapunov candidate
    V = x.dot(best_P.dot(x))

    # Bilinear seaerch for ROA
    Vs, rhos = maximize_rho_bilinear(
        V, x, f, bilinear_iter, V_degree=2, lambda_degree=6)

    # visualize the dynamics
    sys = SymbolicVectorSystem(state=x, dynamics=f)
    fig, ax = plt.subplots(figsize=(10, 20))
    xlim = (-4,4)
    ylim = (-4,4)

    plot_2d_phase_portrait(sys, xlim, ylim, colored=True)
    plt.plot([0], [0], '*', color='red', markersize='10')

    # plot the certified ROA       
    n = 100
    best_V = Vs[-1]
    best_rho = rhos[-1]
    X1 = np.linspace(xlim[0], xlim[1], n)
    X2 = np.linspace(ylim[0], ylim[1], n)
    Z = np.zeros((n,n))

    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            Z[j,i] = best_V.Evaluate({x[0]: x1, x[1]: x2})
    plt.contour(X1, X2, Z, levels=[best_rho], 
                        linestyles='dashed', cmap=plt.get_cmap('autumn'), 
                        linewidths=2)
    plt.show()
    print(f"Best Q:\n{Q}")
    print(f"Best R:\n{R}")
    print(f"Best K:\n{K}")
    print(f"Volume: {best_volume}")
    print(f"Certified that {str(best_V)} < {best_rho} is a ROA")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='bilinear_ROA',
                    description='Maximizes ROA for simple pendulum with bilinear search')
    parser.add_argument('--RL_iter', type=int, default=3)
    parser.add_argument('--bilinear_iter', type=int, default=3)
    args = parser.parse_args()

    optimal_lqr(args.RL_iter, args.bilinear_iter)