import math
import os

import matplotlib.pyplot as plt
import numpy as np

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    LinearQuadraticRegulator,
    MeshcatVisualizer,
    Parser,
    Simulator,
    StartMeshcat,
    Linearize
)
from pydrake.systems.primitives import ConstantVectorSource, MatrixGain, Adder
from pydrake.examples import PendulumPlant

np.random.seed(1)

def cartpole_balancing_example(meshcat):
    def UprightState():
        state = (np.pi, 0)
        return state

    def BalancingLQR():
        LQRbuilder = DiagramBuilder()
        pendulum = LQRbuilder.AddSystem(PendulumPlant())
        context = pendulum.CreateDefaultContext()
        pendulum.get_input_port(0).FixValue(context, [0])
        context.SetContinuousState([np.pi, 0])

        # linearize the system at [pi, 0]
        linearized_pendulum = Linearize(pendulum, context)

        # LQR
        Q = np.diag((5., 1.))
        R = [1.]
        return LinearQuadraticRegulator(linearized_pendulum.A(), linearized_pendulum.B(), Q, R)

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.1)
    parser = Parser(plant)
    parser.AddModelFromFile("/home/adam/workspace/ECE1659/examples/pendulum/Pendulum.urdf")
    plant.Finalize()

    (K, P) = BalancingLQR()
    gain = builder.AddSystem(MatrixGain(-K))
    offset = builder.AddSystem(ConstantVectorSource(K.dot([np.pi, 0])))
    adder = builder.AddSystem(Adder(2, 1))
    
    builder.Connect(plant.get_state_output_port(), gain.get_input_port(0))
    builder.Connect(gain.get_output_port(0), adder.get_input_port(0))
    builder.Connect(offset.get_output_port(0), adder.get_input_port(1))
    builder.Connect(adder.get_output_port(0), plant.get_actuation_input_port())

    # Setup visualization
    meshcat.Delete()
    meshcat.Set2dRenderMode(xmin=-2.5, xmax=2.5, ymin=-1.0, ymax=2.5)
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    diagram = builder.Build()

    # Set up a simulator to run this diagram
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    # Simulate
    simulator.set_target_realtime_rate(1.0)
    duration = 3.0
    rho = 33.25
    for i in range(20):
        while True:
            xbar0 = np.multiply(np.random.randn(2), np.array([1.0, 4.0]))
            if xbar0.dot(P.dot(xbar0)) < rho:
                break
        x0 = UprightState() + xbar0
        context.SetTime(0.0)
        plant.SetPositionsAndVelocities(
            plant_context,
            x0
        )
        simulator.Initialize()
        simulator.AdvanceTo(duration)

if __name__ == '__main__':
    meshcat = StartMeshcat()
    cartpole_balancing_example(meshcat)