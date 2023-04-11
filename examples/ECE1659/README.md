# ECE1659 Final Project

## SOS Optimization for Region of Attraction Analysis and Controller Design
This directory contains all the code required to reproduce the demonstrations from the presentation.

**global_lyapunov.py:** searches for a Lyapunov function to certify the global stability of a closed-loop system of a jet engine.

**local_lyapunov.py:** find the region of attraction (ROA) for an LQR controller applied to an inverted pendulum system using the S-Procedure.

**bilinear_ROA.py:** iteratively grow the ROA for the LQR controller using bilinear search.

**optimal_LQR.py:** uses gradient-free RL (weight perturbation algorithm) to search for LQR controllers with large ROAs for a polynomial system.

**Note:** The RL algorithm was designed solely for the demonstration. Although it works (with enough parameter tuning), there are many limitations to this approach. There are likely better ways to design LQR controllers with large ROAs, but I have no conducted a literature review.

## Running the code
1. To build the required libraries (from the root directory):
```
bazel build examples/ECE1659/...
```
2. To run an example (ex. for global_lyapunov.py):
```
bazel-bin/examples/ECE1659/global_lyapunov
```
## Resources
1. [Structured Semidefinite Programs and Semialgebraic Geometry Methods in Robustness and Optimization](https://www.mit.edu/~parrilo/pubs/files/thesis.pdf)  by [Prof. Pablo Parrilo](https://www.mit.edu/~parrilo/): Chapters 4 and 7
2. [MAE509](https://control.asu.edu/) taught by [Prof. Matthew M. Peet](https://control.asu.edu/): Lectures [16](https://control.asu.edu/Classes/MAE598/598Lecture16.pdf) and [17](https://control.asu.edu/Classes/MAE598/598Lecture17.pdf)
3. [Underactuated Robotics](https://underactuated.csail.mit.edu/) by [Prof. Russ Tedrake](https://groups.csail.mit.edu/locomotion/russt.html): Chapter 9
