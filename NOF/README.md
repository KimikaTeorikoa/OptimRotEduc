# Natural Orbital Optimization

This folder contains the files for solving Natural Orbital Functional calculations with Steepest Gradient Descent, Conjugate Gradients, AdaGrad, RMSProp and ADAM optimization algorithms.

## Comparison of optimization algorithms

This exercise compares the performance of the several optimization algorithms in natural orbital functional theory at two step sizes: 0.02 and 0.002.

An example for the carbon monoxide (CO) molecule is provided, containing the following files:
- 0_NOF-LR002.ipynb and 1_NOF-LR002.ipynb: Contain the code to solve a Natural Orbital Functional using Steepest Gradient Descent, Conjugate Gradients, AdaGrad, RMSProp and ADAM at 0.02 and 0.002 step sizes and write the results into csv files.
- 2_Plot-NOF-LR02.ipynb and 3_Plot-NOF-LR002.ipynb: Read the csv file and plot the energy with respect to the iteration number for all the algorithms.

## Dynamic Step

This exercise compares the performance of a dynamic (scheduled) step size for the several optimization algorithms.

An example for the boron trifluoride (BF3) molecule is provided, containing the following files:
- 0_Dynamic-Step.ipynb: Contain the code to solve a Natural Orbital Functional using Steepest Gradient Descent, Conjugate Gradients, AdaGrad, RMSProp and ADAM at several step sizes and at a scheduled step size and write the results into csv files.
- 1_Plot-Dynamic-Step.ipynb: Read the csv file and plot the energy with respect to the iteration number for all the algorithms.