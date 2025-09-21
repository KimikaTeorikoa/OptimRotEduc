# ⭐ Hartree-Fock

This folder contains the files for solving Hartree-Fock with Steepest Gradient Descent, Conjugate Gradients, AdaGrad, RMSProp and ADAM optimization algorithms.

## 🏃 Algorithm coding and Step size tunning

This exercise explore the effect of the step size (alpha) on the different algorithms.

An example for the water (H2O) molecule is provided, containing the following files:
- **0_Opt-Algorithm-Coding.ipynb**: This first notebook present an example for doing Hartree-Fock using steepest descent. The task is to use this example to complete the code in myCode/minimization.py for the other algorithms: Conjugate Gradients, AdaGrad, RMSProp and ADAM.
- **1_Opt-Step-Size.ipynb**: Contains the code to solve Hartree-Fock using Steepest Gradient Descent, Conjugate Gradients, AdaGrad, RMSProp and ADAM with several step sizes (learning rates) and write the results into csv files.
- **2_Plot-ADAGrad.ipynb**: Read the csv file and plot the energy with respect to the iteration number when the Adaptive Gradient method is used.
- **3_Plot-RMSProp.ipynb**: Read the csv file and plot the energy with respect to the iteration number when the Root Mean Square Propagation method is used.
- **4_Plot-ADAM.ipynb**: Read the csv file and plot the energy with respect to the iteration number when the Adaptive Momentum method is used.

## 🚗 Comparison of optimization algorithms

This exercise compares the performance of the several optimization algorithms at their best step size.

An example for the boron trifluoride (BF3) molecule is provided, containing the following files:
- **0_HF-Opt-Algorithm-Best-Step-Size.ipynb**: Contains the code to solve Hartree-Fock using Steepest Gradient Descent, Conjugate Gradients, AdaGrad, RMSProp and ADAM at their best identified step sizes and write the results into csv files.
- **1_HF-Plot-Opt-Method-Best-Step-Size.ipynb**: Read the csv file and plot the energy with respect to the iteration number for all the algorithms.