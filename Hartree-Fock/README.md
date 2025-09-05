# Hartree-Fock

This folder contains the files for solving Hartree-Fock with Steepest gradient descent, Conjugate Gradients, AdaGrad, RMSProp and ADAM.


## Algorithm coding and Step size Tunning

This exercise experiment the effect of the step size (alpha) on the different algorithms.

An example for the water (H2O) molecule is provided, containing the following files:
- 0_H2O.ipynb: Contain the code to solve Hartree-Fock using Steepest gradient descent, Conjugate Gradients, AdaGrad, RMSProp and ADAM with several step sizes (leraning rates) and write the results into csv files.
- 1_Plot-ADAGrad.ipynb: Read the csv file and plot the energy with respect to the iteration number when the Adaptive Gradient method is used.
- 2_Plot-RMSProp.ipynb: Read the csv file and plot the energy with respect to the iteration number when the Root Mean Square Propagation method is used.
- 3_Plot-ADAM.ipynb: Read the csv file and plot the energy with respect to the iteration number when the Adaptive Momentum method is used.

## Comparison of optimization algorithms

This exercise compares the performance of the several optimization algorithms at its best step size.

An example for the boron trifluoride (BF3) molecule is provided, containing the following files:
- 0_BF3.ipynb: Contain the code to solve Hartree-Fock using Steepest gradient descent, Conjugate Gradients, AdaGrad, RMSProp and ADAM at their best identified step sizes and write the results into csv files.
- 1_Plot-BF3.ipynb: Read the csv file and plot the energy with respect to the iteration number for all the algorithms.
