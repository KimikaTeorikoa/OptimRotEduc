# ⭐ Hartree-Fock

This folder contains the files for solving Hartree-Fock with Steepest Gradient Descent, Conjugate Gradients, AdaGrad, RMSProp and ADAM optimization algorithms. These notebooks were used to generate the results of the HF section of the manuscript.

## 🏃 Example case
- **0_Test_Case.ipynb**: This first notebook presents an example for doing a Hartree-Fock energy calculation using the steepest descent algorithm to optimize the orbitals. Once verified that SD works, based on the provided myCode/minimization.py, one should code the  optimization algorithms of interest, in our case: Conjugate Gradients, AdaGrad, RMSProp and ADAM. Once these algorithms are coded, one can continue with the next step.

## 🏃 Algorithm coding and Step size effect (A_StepTunning Folder)

This exercise explore the effect of the step size (alpha) on the different algorithms.

An example for the H2O and CO2 molecules is provided, containing the following files:
- **1_HF_calcs.ipynb**: Contains the code to solve Hartree-Fock using Steepest Gradient Descent, Conjugate Gradients, AdaGrad, RMSProp and ADAM with several step sizes (learning rates) and write the results into csv files. This notebook generates the data plotted in Figure 1 (and the figure in supporting information) of the manuscript.
- **2_Plot.ipynb**: Reads the csv files and plots the energy with respect to the iteration number for a given method. This notebook was used to plot the data shown in Figure 1 (and the figure in supporting information) of the manuscript.

Both notebooks are prepared to run calculations and plot the results for H2O and CO2 in a single run.

## 🚗 Comparison of optimization algorithms (B_MethodComparison folder)

This exercise compares the performance of the several optimization algorithms at their best step size, according to the results of the previous folder.

An example for the BF3, Al(OH)3 and CHCl3 molecules is provided, containing the following files:
- **0_HF_calcs.ipynb**: Contains the code to solve Hartree-Fock using Steepest Gradient Descent, Conjugate Gradients, AdaGrad, RMSProp and ADAM at their best identified step sizes and write the results into csv files. This notebook generates the data plotted in Figure 2 of the manuscript.
- **1_Plot.ipynb**: Read the csv file and plot the energy with respect to the iteration number for all the algorithms.
This notebook was used to plot the data shown in Figure 2 of the manuscript.
