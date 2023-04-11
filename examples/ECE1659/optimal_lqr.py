import math
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import argparse

from plotting_utils import *
from pendulum_utils import *

from pydrake.all import (
    MathematicalProgram,
    SymbolicVectorSystem,
    ToLatex
)

import sys, os

np.random.seed(736) # Just for demo

# Disable
def block_print():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enable_print():
    sys.stdout = sys.__stdout__

def compute_ROA_volume(V, x, rho):
    V_normalized = Polynomial(V / rho)

    prog = MathematicalProgram()
    prog.AddIndeterminates(x)

    (V_prog, P) = prog.NewSosPolynomial(Variables(x), 2)
    prog.AddEqualityConstraintBetweenPolynomials(V_normalized, V_prog)
    result = Solve(prog)
    return np.trace(np.linalg.inv(result.GetSolution(P)[0:2,0:2]))

def compute_cost(volume, Q, R):
    param_Q_ratio_term = 0.1
    param_QR_ratio_term = 0.01
    # param_QR_ratio_term = 0.

    return -volume \
           + param_Q_ratio_term * max(Q[0,0]/Q[1,1], Q[1,1]/Q[0,0]) \
           + param_QR_ratio_term * math.sqrt(Q[0,0]**2 + Q[1,1]**2) / R[0]


def optimal_lqr(RL_iter=10, bilinear_iter = 3, save=0):
    base_prog = MathematicalProgram()
    x = base_prog.NewIndeterminates(2, "x")

    def animate(i):
        # Check if drawing end frames
        if i > RL_iter:
            i = RL_iter + 1

        plt.clf()
        # Closed loop system
        u = -Ks[i].dot(x)
        f = [-x[0] - 2*x[1]**2 + u,
            x[1] + x[0]*x[1] + 2*x[1]**3 + u]
        sys = SymbolicVectorSystem(state=x, dynamics=f)

        # visualize the dynamics
        xlim = (-10, 10)
        ylim = (-10, 10)

        plot_2d_phase_portrait(sys, xlim, ylim, colored=True)
        plt.plot([0], [0], '*', color='red', markersize='10')
        if i <= RL_iter:
            plt.title(f"Certified ROA for Iteration {i}")
        else:
            plt.title("Final Certified ROA")

        # plot the certified ROA       
        n = 100
        X1 = np.linspace(xlim[0], xlim[1], n)
        X2 = np.linspace(ylim[0], ylim[1], n)
        Z = np.zeros((n,n))

        V = Vs[i]
        rho = rhos[i]
        for i, x1 in enumerate(X1):
            for j, x2 in enumerate(X2):
                Z[j,i] = V.Evaluate({x[0]: x1, x[1]: x2})
        plt.contour(X1, X2, Z, levels=[rho], 
                            linestyles='dashed', cmap=plt.get_cmap('autumn'), 
                            linewidths=2)

    # Setup linearized system
    A = np.array([[-1, 0], [0, 1]])
    B = np.ones((2,1))
    Q = np.diag((1., 1.))
    R = [1]

    # learning parameters
    alpha = 0.1
    baseline_cost = float('inf')
    best_volume = -1
    best_Q = Q
    best_R = R
    best_cost = float('inf')

    # for training plots
    Q1s = []
    Q2s = []
    Rs = []
    volumes = []
    costs = []
    best_V = None
    best_rho = -1
    best_K = None
    
    # for ROA animations
    Vs = []
    rhos = []
    Ks = []

    # Search for LQR controller with largest ROA using weight perturbation
    for iter in range(RL_iter+1):
        """ Compute the new baseline volume and plot it """
        (K, P) = LinearQuadraticRegulator(A, B, Q, R)

        # Define the SOS program
        prog = MathematicalProgram()
        prog.AddIndeterminates(x)

        # Closed loop dyanmics
        u = -K.dot(x)
        f = [-x[0] - 2*x[1]**2 + u,
            x[1] + x[0]*x[1] + 2*x[1]**3 + u]
        
        # Use cost-to-go as the Lyapunov candidate
        V = x.dot(P.dot(x))

        # Bilinear seaerch for ROA
        block_print()
        V_, rho_ = maximize_rho_bilinear(
            V, x, f, bilinear_iter, V_degree=2, lambda_degree=6)
        enable_print()
        if len(V_) == 0:
            print("Failed to certify any ROA with current Q")
            return
        
        baseline_volume = compute_ROA_volume(V_[-1], x, rho_[-1])
        baseline_cost = compute_cost(baseline_volume, Q, R)

        if baseline_cost < best_cost:
            best_cost = baseline_cost
            best_volume = baseline_volume
            best_Q = Q
            best_R = R
            best_V = V_[-1]
            best_rho = rho_[-1]
            best_K = K

        # Print
        print(f"Iteration {iter}")
        print("------------------------------------------------------------")
        print(f"Q: diag({Q[0,0]}, {Q[1,1]})")
        print(f"R: {R}")
        print(f"Cost: {baseline_cost}")
        print(f"Volume: {baseline_volume}")
        print("\n")
        
        # Store training data for plotting
        Q1s.append(Q[0,0])
        Q2s.append(Q[1,1])
        Rs.append(R[0])
        volumes.append(baseline_volume)
        costs.append(baseline_cost)

        # Store data for visualization
        Vs.append(V_[-1])
        rhos.append(rho_[-1])
        Ks.append(K)

        if iter == RL_iter:
            break

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
            new_R = R + dR
            # check that new_R is PD
            if new_R > 0:
                break

        # Evalute volume of new_Q
        (K, P) = LinearQuadraticRegulator(A, B, new_Q, new_R)

        # Define the SOS program
        prog = MathematicalProgram()
        prog.AddIndeterminates(x)

        # Closed loop dyanmics
        u = -K.dot(x)
        f = [-x[0] - 2*x[1]**2 + u,
            x[1] + x[0]*x[1] + 2*x[1]**3 + u]
        
        # Use cost-to-go as the Lyapunov candidate
        V = x.dot(P.dot(x))

        # Bilinear seaerch for ROA
        block_print()
        V_, rho_ = maximize_rho_bilinear(
            V, x, f, bilinear_iter, V_degree=2, lambda_degree=6)
        enable_print()

        if len(V_) == 0:
            print("Failed to certify any ROA with new_Q")
            return
        
        # Update Q
        new_volume = compute_ROA_volume(V_[-1], x, rho_[-1])
        new_cost = compute_cost(new_volume, new_Q, new_R)
        alpha_temp = alpha
        while True:
            Q_perturbed = Q - alpha_temp*(new_cost-baseline_cost)*dQ
            R_perturbed = R - alpha_temp*(new_cost-baseline_cost)*dR
            if Q_perturbed[0][0] >= 0 and Q_perturbed[1][1] >= 0 and R_perturbed[0] > 0:
                Q = Q_perturbed
                R = R_perturbed
                break
            else:
                alpha_temp /= 2.

                
    """ RL Search Complete. Plot best V, rho, K """
    Vs.append(best_V)
    rhos.append(best_rho)
    Ks.append(best_K)

    print(f"Best Q:\n{best_Q}")
    print(f"Best R:\n{best_R}")
    print(f"Best K:\n{Ks[-1]}")
    print(f"Certified that {ToLatex((Polynomial(Vs[-1])/rhos[-1]).ToExpression(), 6)} < {1} is a ROA")

    fig, ax = plt.subplots(figsize=(20, 10))   
       
    if save:
        print("Preparing animation...")
        last_frame_duration = 5
        num_frames = RL_iter + 2 + last_frame_duration
        anim = ani.FuncAnimation(fig, animate, frames=num_frames, interval=1000)
        anim.save("examples/ECE1659/animations/lqr_search.mp4")
    else:
        for i in range(RL_iter+2):
            animate(i)
            if i == RL_iter+1:
                plt.pause(5)
            else:
                plt.pause(0.5)
    plt.show()

    return best_volume


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='bilinear_ROA',
                    description='Maximizes ROA for simple pendulum with bilinear search')
    parser.add_argument('--RL_iter', type=int, default=20)
    parser.add_argument('--bilinear_iter', type=int, default=3)
    parser.add_argument('--save', type=int, default=0)
    args = parser.parse_args()

    volume = optimal_lqr(args.RL_iter, args.bilinear_iter, args.save)