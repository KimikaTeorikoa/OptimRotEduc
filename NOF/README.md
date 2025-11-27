# 🌟 Natural Orbital Functionals

This folder contains the files for solving Natural Orbital Functional calculations with Steepest Gradient Descent, Conjugate Gradients, AdaGrad, RMSProp and ADAM optimization algorithms.

## 📊 Comparison of optimization algorithms

This exercise compares the performance of the several optimization algorithms in natural orbital functional theory at two step sizes: 0.02 and 0.002.

Examples for the CO, CO2 and HNO3 molecules are provided, containing the following files:
- **0_NOF.ipynb** : Contains the code to solve a Natural Orbital Functional using Steepest Gradient Descent, Conjugate Gradients, AdaGrad, RMSProp and ADAM at 0.02 and 0.002 step sizes and write the results into csv files.  This notebook was used to generate the results shown in Figure 3 of the provided manuscript. 
- **1_Plot.ipynb** Reads the corresponding csv files (molecules calculated with different step size)  and plot the energy with respect to the iteration number for all the algorithms. This notebook was used to plot the results obtained with the previous notebook to gemerate in Figure 3 of the provided manuscript.

## 🔥 Dynamic Step

This exercise compares the performance of a dynamic (scheduled) step size for the several optimization algorithms.

Examples for the BF3, O3 and HNO3 molecules are provided, containing the following files:
- **0_NOF_calcs.ipynb**: Contain the code to solve a Natural Orbital Functional using Steepest Gradient Descent, Conjugate Gradients, AdaGrad, RMSProp and ADAM at several step sizes and at a scheduled step size and write the results into csv files.  This notebook was used to generate the results shown in Figure 4 of the provided manuscript. Executes calculations for the three molecules in one run.
- **1_Plot.ipynb**: Reads the csv file and plot the energy with respect to the iteration number for all the algorithms.
This notebook was used to plot the results obtained with the previous notebook to gemerate in Figure 4 of the provided manuscript. Plots all the results in one run.

