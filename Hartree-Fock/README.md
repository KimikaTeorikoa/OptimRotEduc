# ⭐ Hartree-Fock

This folder contains the files for solving Hartree-Fock with Steepest Gradient Descent, Conjugate Gradients, AdaGrad, RMSProp and ADAM optimization algorithms. This notebooks where used to generate the results of the HF section of the manuscript.

## 🏃 Example case
- **0_Test_Case.ipynb**: This first notebook present an example for doing a Hartree-Fock energy calculation using steepest descent algorithm to optimize the orbitals. Once verified that SD works, based on the provided myCode/minimization.py, pne should 
code the  optimization algorithms of interest, in our case: Conjugate Gradients, AdaGrad, RMSProp and ADAM. Once these algorithms are coded, one can continue with the next step.

## 🏃 Algorithm coding and Step size effect (A_StepTunning Folder)

This exercise explore the effect of the step size (alpha) on the different algorithms.

An example for the water (H2O) molecule is provided, containing the following files:
- **1_HF_calcs.ipynb**: Contains the code to solve Hartree-Fock using Steepest Gradient Descent, Conjugate Gradients, AdaGrad, RMSProp and ADAM with several step sizes (learning rates) and write the results into csv files. This notebook generates the data ploted in Figure 1 of the manuscript.
- **2_Plot.ipynb**: Read the csv file and plot the energy with respect to the iteration number for a given method. This notebook was used to plot the 
data shown in Figure 1 of the manuscript.

Once the H2O example run, one can substitute H2O by CO2, and run the notebook. In **2_Plot** notebook, in order to adjust the axis accordingly to the selected molecule, the corresponding code must be comented/uncommented for the selected molecule

## 🚗 Comparison of optimization algorithms (B_MethodComparison folder)

This exercise compares the performance of the several optimization algorithms at their best step size, according to the results of the previous folder.

An example for the boron trifluoride (BF3) molecule is provided, containing the following files:
- **0_HF_calcs.ipynb**: Contains the code to solve Hartree-Fock using Steepest Gradient Descent, Conjugate Gradients, AdaGrad, RMSProp and ADAM at their best identified step sizes and write the results into csv files. This notebook generates the data ploted in Figure 1 of the manuscript.
- **1_Plot.ipynb**: Read the csv file and plot the energy with respect to the iteration number for all the algorithms.
This notebook was used to plot the data shown in Figure 2 of the manuscript.