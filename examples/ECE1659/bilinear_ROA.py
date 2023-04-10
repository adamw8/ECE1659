import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import argparse

from plotting_utils import *
from pendulum_utils import *

def bilinear_ROA(bilinear_iter = 3, save=0):
    # Setup system
    builder = DiagramBuilder()
    pendulum = builder.AddSystem(PendulumPlant())
    context = pendulum.CreateDefaultContext()
    pendulum.get_input_port(0).FixValue(context, [0])
    context.SetContinuousState([np.pi, 0])

    # linearize the system at [pi, 0]
    linearized_pendulum = Linearize(pendulum, context)

    # LQR
    Q = np.diag((5, 1.))
    R = [1.]
    (K, P) = LinearQuadraticRegulator(linearized_pendulum.A(),
                                      linearized_pendulum.B(),
                                      Q, R)

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

    # Bilinear seaerch for ROA
    Vs, rhos = maximize_rho_bilinear(V, xbar, f, bilinear_iter, lambda_degree=2)

    def animate(i):
        plt.clf()

        # visualize the dynamics
        xlim = (-2.1*np.pi,2.1*np.pi)
        ylim = (-25,25)

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
        
        if i > 0:
            # plot the certified ROA
            n = 100
            X1 = np.linspace(xlim[0], xlim[1], n)
            X2 = np.linspace(ylim[0], ylim[1], n)
            Z = np.zeros((n,n))

            V = Vs[i-1]
            rho = rhos[i-1]
            for i, x1 in enumerate(X1):
                for j, x2 in enumerate(X2):
                    Z[j,i] = V.Evaluate({xbar[0]: x1-np.pi, xbar[1]: x2})
            plt.contour(X1, X2, Z, levels=[rho], 
                                linestyles='dashed', cmap=plt.get_cmap('autumn'), 
                                linewidths=2)
    
    fig, ax = plt.subplots(figsize=(20, 10))
    if save:
        print("Preparing animation...")
        last_frame_duration = 5
        anim = ani.FuncAnimation(fig, animate, frames=len(Vs), interval=1000)
        anim.save("examples/ECE1659/animations/bilinear_ROA.mp4")
    else:
        for i in range(len(Vs)+1):
            animate(i)
            if i == len(Vs):
                plt.pause(5)
            else:
                plt.pause(0.5)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='bilinear_ROA',
                    description='Maximizes ROA for simple pendulum with bilinear search')
    parser.add_argument('--bilinear_iter', type=int, default=10)
    parser.add_argument('--save', type=int, default=0)
    args = parser.parse_args()

    bilinear_ROA(args.bilinear_iter, args.save)
    